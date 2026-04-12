"""Tests for unified optimizer interface."""

import math
import pytest
import torch

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_trev.model import TensorRingModel
from qiskit_trev.optimization import (
    OptimizationResult,
    CMAESOptimizer,
    GradientOptimizer,
)


def _make_ry_z_model():
    """Simple 1-param model: RY(theta)|0> with Z observable."""
    qc = QuantumCircuit(1)
    qc.ry(0.0, 0)
    op = SparsePauliOp.from_list([("Z", 1.0)])
    return TensorRingModel(qc, op, rank=1, device="cpu")


def _make_2param_model():
    """2-param model with ZI+IZ Hamiltonian."""
    qc = QuantumCircuit(2)
    qc.ry(0.0, 0)
    qc.ry(0.0, 1)
    op = SparsePauliOp.from_list([("ZI", 0.5), ("IZ", 0.5)])
    return TensorRingModel(qc, op, rank=4, device="cpu")


# ============================================================
# OptimizationResult
# ============================================================

class TestOptimizationResult:

    def test_fields(self):
        r = OptimizationResult(
            params=torch.tensor([1.0]),
            cost=0.5,
            cost_history=[1.0, 0.7, 0.5],
            num_iterations=3,
        )
        assert r.params.shape == (1,)
        assert r.cost == 0.5
        assert len(r.cost_history) == 3
        assert r.num_iterations == 3


# ============================================================
# CMAESOptimizer
# ============================================================

class TestCMAESOptimizer:

    def test_minimize_returns_result(self):
        model = _make_ry_z_model()
        opt = CMAESOptimizer(sigma=0.5, pop_size=8)
        result = opt.minimize(model, torch.tensor([0.5]), max_iter=5)
        assert isinstance(result, OptimizationResult)
        assert result.params.shape == (1,)
        assert len(result.cost_history) == 5
        assert result.num_iterations == 5

    def test_converges_ry_z(self):
        """Should find theta~pi where cos(theta)=-1."""
        model = _make_ry_z_model()
        opt = CMAESOptimizer(sigma=0.5, pop_size=8)
        result = opt.minimize(model, torch.tensor([0.5]), max_iter=30)
        assert result.cost < -0.8

    def test_converges_2param(self):
        model = _make_2param_model()
        opt = CMAESOptimizer(sigma=0.5, pop_size=10)
        result = opt.minimize(model, torch.tensor([0.0, 0.0]), max_iter=30)
        assert result.cost < -0.5

    def test_custom_pop_size(self):
        model = _make_ry_z_model()
        opt = CMAESOptimizer(sigma=1.0, pop_size=20)
        result = opt.minimize(model, torch.tensor([0.0]), max_iter=3)
        assert len(result.cost_history) == 3


# ============================================================
# GradientOptimizer
# ============================================================

class TestGradientOptimizer:

    def test_minimize_returns_result(self):
        model = _make_ry_z_model()
        opt = GradientOptimizer(lr=0.1)
        result = opt.minimize(model, torch.tensor([0.5]), max_iter=5)
        assert isinstance(result, OptimizationResult)
        assert result.params.shape == (1,)
        assert len(result.cost_history) == 5

    def test_converges_ry_z(self):
        """Should find theta~pi where cos(theta)=-1."""
        model = _make_ry_z_model()
        opt = GradientOptimizer(lr=0.1)
        result = opt.minimize(model, torch.tensor([0.5]), max_iter=50)
        assert result.cost < -0.8

    def test_converges_2param(self):
        model = _make_2param_model()
        opt = GradientOptimizer(lr=0.1)
        result = opt.minimize(model, torch.tensor([0.5, 0.5]), max_iter=50)
        assert result.cost < -0.5

    def test_adam_optimizer(self):
        """Use Adam instead of default SGD."""
        model = _make_ry_z_model()
        opt = GradientOptimizer(lr=0.1, optimizer_cls="adam")
        result = opt.minimize(model, torch.tensor([0.5]), max_iter=50)
        assert result.cost < -0.8

    def test_custom_chunk_size(self):
        model = _make_2param_model()
        opt = GradientOptimizer(lr=0.1, chunk_size=1)
        result = opt.minimize(model, torch.tensor([0.0, 0.0]), max_iter=5)
        assert len(result.cost_history) == 5


# ============================================================
# Both optimizers share same interface
# ============================================================

class TestUnifiedInterface:

    @pytest.mark.parametrize("opt_cls,kwargs", [
        (CMAESOptimizer, dict(sigma=0.5, pop_size=8)),
        (GradientOptimizer, dict(lr=0.1)),
    ])
    def test_same_api(self, opt_cls, kwargs):
        model = _make_ry_z_model()
        opt = opt_cls(**kwargs)
        result = opt.minimize(model, torch.tensor([0.5]), max_iter=10)
        assert hasattr(result, 'params')
        assert hasattr(result, 'cost')
        assert hasattr(result, 'cost_history')
        assert hasattr(result, 'num_iterations')
