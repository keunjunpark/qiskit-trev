"""Tests for batched expectation value and BatchParameterShiftGradient."""

import math
import pytest
import torch

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_trev.model import TensorRingModel
from qiskit_trev.gradient import BatchParameterShiftGradient


class TestBatchParameterShiftGradient:

    def test_gradient_ry_z(self):
        """d/dtheta cos(theta) = -sin(theta)."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        grad_fn = BatchParameterShiftGradient(model)

        theta = torch.tensor([0.7])
        grad = grad_fn(theta)
        assert abs(grad[0].item() + math.sin(0.7)) < 1e-3

    def test_gradient_matches_sequential(self):
        """Batch gradient should match sequential parameter shift."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        qc.cx(0, 1)
        op = SparsePauliOp.from_list([("ZZ", 1.0)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")

        theta = torch.tensor([0.5, 1.0])
        grad_seq = model.parameter_shift_grad(theta)

        grad_fn = BatchParameterShiftGradient(model)
        grad_batch = grad_fn(theta)

        assert torch.allclose(grad_batch, grad_seq, atol=1e-3)

    def test_gradient_shape(self):
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.rx(0.0, 1)
        op = SparsePauliOp.from_list([("ZI", 1.0)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")
        grad_fn = BatchParameterShiftGradient(model)
        grad = grad_fn(torch.tensor([0.3, 0.7]))
        assert grad.shape == (2,)

    def test_gradient_at_zero(self):
        """At theta=0, d/dtheta cos(theta) = 0."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        grad_fn = BatchParameterShiftGradient(model)
        grad = grad_fn(torch.tensor([0.0]))
        assert abs(grad[0].item()) < 1e-3

    def test_chunk_size(self):
        """Chunked gradient should match full batch."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        op = SparsePauliOp.from_list([("ZZ", 1.0)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")

        theta = torch.tensor([0.5, 1.0])
        grad_full = BatchParameterShiftGradient(model, chunk_size=10)(theta)
        grad_chunked = BatchParameterShiftGradient(model, chunk_size=1)(theta)
        assert torch.allclose(grad_full, grad_chunked, atol=1e-4)

    def test_no_params(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        grad_fn = BatchParameterShiftGradient(model)
        grad = grad_fn(torch.tensor([]))
        assert grad.shape == (0,)

    def test_4_params(self):
        """4-parameter circuit gradient."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        qc.cx(0, 1)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        op = SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")

        theta = torch.tensor([0.3, 0.7, 1.1, 0.2])
        grad_seq = model.parameter_shift_grad(theta)
        grad_batch = BatchParameterShiftGradient(model, chunk_size=2)(theta)
        assert torch.allclose(grad_batch, grad_seq, atol=1e-3)
