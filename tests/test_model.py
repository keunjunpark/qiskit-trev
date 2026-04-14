"""Tests for TensorRingModel (PyTorch nn.Module)."""

import math
import pytest
import torch
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_trev.model import TensorRingModel


class TestForward:

    def test_Z_on_zero(self):
        """<0|Z|0> = 1.0."""
        qc = QuantumCircuit(1)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        ev = model(torch.tensor([]))
        assert abs(ev.item() - 1.0) < 1e-5

    def test_ZZ_on_bell(self):
        """<Bell|ZZ|Bell> = 1.0."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        op = SparsePauliOp.from_list([("ZZ", 1.0)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")
        ev = model(torch.tensor([]))
        assert abs(ev.item() - 1.0) < 1e-4

    def test_parameterized_circuit(self):
        """RY(theta)|0> with Z observable: <Z> = cos(theta)."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)  # placeholder, will be overridden by params
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        theta = torch.tensor([0.5])
        ev = model(theta)
        assert abs(ev.item() - math.cos(0.5)) < 1e-4

    def test_multi_term_hamiltonian(self):
        """H = ZI + 0.5*IZ on |00>: 1.0 + 0.5 = 1.5."""
        qc = QuantumCircuit(2)
        op = SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 0.5)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")
        ev = model(torch.tensor([]))
        assert abs(ev.item() - 1.5) < 1e-5

    def test_returns_tensor(self):
        qc = QuantumCircuit(1)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        ev = model(torch.tensor([]))
        assert isinstance(ev, torch.Tensor)
        assert ev.dim() == 0  # scalar

    def test_x_on_plus(self):
        """<+|X|+> = 1.0."""
        qc = QuantumCircuit(1)
        qc.h(0)
        op = SparsePauliOp.from_list([("X", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        ev = model(torch.tensor([]))
        assert abs(ev.item() - 1.0) < 1e-4


class TestParameterShiftGrad:

    def test_gradient_ry_z(self):
        """d/dtheta <0|RY(-theta)Z RY(theta)|0> = -sin(theta)."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        theta = torch.tensor([0.7])
        grad = model.parameter_shift_grad(theta)
        expected = -math.sin(0.7)
        assert abs(grad[0].item() - expected) < 1e-3

    def test_gradient_shape(self):
        """Gradient should have same shape as params."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        op = SparsePauliOp.from_list([("ZZ", 1.0)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")
        theta = torch.tensor([0.5, 1.0])
        grad = model.parameter_shift_grad(theta)
        assert grad.shape == (2,)

    def test_gradient_at_extremum(self):
        """At theta=0, d/dtheta cos(theta) = 0."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        theta = torch.tensor([0.0])
        grad = model.parameter_shift_grad(theta)
        assert abs(grad[0].item()) < 1e-3

    def test_gradient_at_pi_half(self):
        """At theta=pi/2, d/dtheta cos(theta) = -1."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        theta = torch.tensor([math.pi / 2])
        grad = model.parameter_shift_grad(theta)
        assert abs(grad[0].item() + 1.0) < 1e-3

    def test_no_params_empty_grad(self):
        """Circuit with no parameters: gradient is empty."""
        qc = QuantumCircuit(1)
        qc.h(0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        theta = torch.tensor([])
        grad = model.parameter_shift_grad(theta)
        assert grad.shape == (0,)


class TestNNModule:

    def test_is_nn_module(self):
        qc = QuantumCircuit(1)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        assert isinstance(model, torch.nn.Module)


class TestNonZIBatchForward:

    def test_batch_forward_non_zi_hamiltonian(self):
        """evaluate_batch with non-ZI Hamiltonian uses ev_full path (model.py line 128)."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        op = SparsePauliOp.from_list([("XX", 1.0)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")

        params_batch = torch.tensor([[0.5, 1.0], [0.3, 0.7]])
        evs = model.evaluate_batch(params_batch)
        assert evs.shape == (2,)
        assert torch.isfinite(evs).all()


class TestParameterShiftGradBatch:

    def test_no_params_returns_empty(self):
        """parameter_shift_grad_batch with P=0 returns empty tensor (model.py line 191)."""
        qc = QuantumCircuit(1)
        qc.h(0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")

        # Empty params batch (no parameterized gates)
        params_batch = torch.zeros(3, 0)  # (N=3, P=0)
        result = model.parameter_shift_grad_batch(params_batch)
        assert result.shape == (3, 0)


class TestBuildGatesFallback:

    def test_build_gates_insufficient_values(self):
        """_build_gates falls back to original params when count insufficient (line 69)."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        op = SparsePauliOp.from_list([("ZZ", 1.0)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")

        # Only provide 1 value, but circuit has 2 params
        result = model._build_gates(torch.tensor([1.5]))
        assert result[0].params == (1.5,)
        assert result[1].params == (0.0,)  # fallback to original
