"""Tests for batched parameter_shift_grad and parameter_shift_grad_batch."""

import math
import pytest
import torch

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_trev.model import TensorRingModel


def _make_ry_model(n_qubits=1, rank=1):
    """Single RY(theta) per qubit with Z observable."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(0.0, i)
    pauli = "Z" * n_qubits
    op = SparsePauliOp.from_list([(pauli, 1.0)])
    return TensorRingModel(qc, op, rank=rank, device="cpu")


class TestBatchedParameterShiftGrad:
    """parameter_shift_grad should produce correct results after batching rewrite."""

    def test_ry_z_gradient(self):
        """d/dtheta cos(theta) = -sin(theta)."""
        model = _make_ry_model()
        theta = torch.tensor([0.7])
        grad = model.parameter_shift_grad(theta)
        assert abs(grad[0].item() + math.sin(0.7)) < 1e-3

    def test_gradient_at_zero(self):
        """At theta=0, d/dtheta cos(theta) = 0."""
        model = _make_ry_model()
        grad = model.parameter_shift_grad(torch.tensor([0.0]))
        assert abs(grad[0].item()) < 1e-3

    def test_gradient_at_pi_half(self):
        """At theta=pi/2, d/dtheta cos(theta) = -1."""
        model = _make_ry_model()
        grad = model.parameter_shift_grad(torch.tensor([math.pi / 2]))
        assert abs(grad[0].item() + 1.0) < 1e-3

    def test_multi_param_shape(self):
        """Gradient shape = (P,)."""
        model = _make_ry_model(n_qubits=2, rank=4)
        grad = model.parameter_shift_grad(torch.tensor([0.5, 1.0]))
        assert grad.shape == (2,)

    def test_no_params(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        grad = model.parameter_shift_grad(torch.tensor([]))
        assert grad.shape == (0,)

    def test_matches_finite_difference(self):
        """Batched grad should approximate finite difference."""
        model = _make_ry_model(n_qubits=2, rank=4)
        theta = torch.tensor([0.3, 1.2])
        grad = model.parameter_shift_grad(theta)

        eps = 1e-5
        for i in range(len(theta)):
            t_plus = theta.clone()
            t_plus[i] += eps
            t_minus = theta.clone()
            t_minus[i] -= eps
            fd = (model(t_plus).item() - model(t_minus).item()) / (2 * eps)
            assert abs(grad[i].item() - fd) < 5e-3

    def test_batch_size_parameter(self):
        """batch_size should not affect results."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        qc.cx(0, 1)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        op = SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")
        theta = torch.tensor([0.3, 0.7, 1.1, 0.2])

        grad_full = model.parameter_shift_grad(theta)
        grad_chunked = model.parameter_shift_grad(theta, batch_size=2)

        torch.testing.assert_close(grad_full, grad_chunked, atol=1e-4, rtol=1e-4)


class TestParameterShiftGradBatch:
    """parameter_shift_grad_batch computes gradients for N samples at once."""

    def test_single_sample_matches_grad(self):
        """One sample should match parameter_shift_grad."""
        model = _make_ry_model(n_qubits=2, rank=4)
        theta = torch.tensor([0.5, 1.0])

        grad_single = model.parameter_shift_grad(theta)
        grad_batch = model.parameter_shift_grad_batch(theta.unsqueeze(0))

        assert grad_batch.shape == (1, 2)
        torch.testing.assert_close(grad_batch[0], grad_single, atol=1e-4, rtol=1e-4)

    def test_multiple_samples(self):
        """N samples should each match individual parameter_shift_grad calls."""
        model = _make_ry_model(n_qubits=2, rank=4)
        params_batch = torch.tensor([
            [0.3, 0.7],
            [1.0, 0.5],
            [0.1, 2.0],
        ])

        grad_batch = model.parameter_shift_grad_batch(params_batch)
        assert grad_batch.shape == (3, 2)

        for i in range(3):
            grad_single = model.parameter_shift_grad(params_batch[i])
            torch.testing.assert_close(
                grad_batch[i], grad_single, atol=1e-4, rtol=1e-4
            )

    def test_batch_size_parameter(self):
        """batch_size should not affect results."""
        model = _make_ry_model(n_qubits=2, rank=4)
        params_batch = torch.tensor([
            [0.3, 0.7],
            [1.0, 0.5],
        ])

        grad_full = model.parameter_shift_grad_batch(params_batch)
        grad_chunked = model.parameter_shift_grad_batch(params_batch, batch_size=2)

        torch.testing.assert_close(grad_full, grad_chunked, atol=1e-4, rtol=1e-4)

    def test_output_shape(self):
        """Output is (N, P)."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        qc.cx(0, 1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("ZZ", 1.0)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")

        params_batch = torch.randn(5, 3)
        grad = model.parameter_shift_grad_batch(params_batch)
        assert grad.shape == (5, 3)
