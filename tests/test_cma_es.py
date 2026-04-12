"""Tests for CMA-ES optimizer."""

import math
import pytest
import torch

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_trev.model import TensorRingModel
from qiskit_trev.optimization.cma_es import CMAES, minimize_cma_es


class TestCMAESInit:

    def test_default_pop_size(self):
        """Default population size = 4 + floor(3*ln(n))."""
        cma = CMAES()
        s = cma._init_state(10, torch.device("cpu"))
        expected = 4 + int(3 * math.log(10))
        assert s['lam'] == expected

    def test_custom_pop_size(self):
        cma = CMAES(pop_size=20)
        s = cma._init_state(10, torch.device("cpu"))
        assert s['lam'] == 20

    def test_weights_sum_to_one(self):
        cma = CMAES()
        s = cma._init_state(10, torch.device("cpu"))
        assert abs(s['weights'].sum().item() - 1.0) < 1e-10

    def test_weights_are_decreasing(self):
        cma = CMAES()
        s = cma._init_state(10, torch.device("cpu"))
        w = s['weights']
        for i in range(len(w) - 1):
            assert w[i] >= w[i + 1]

    def test_state_shapes(self):
        cma = CMAES()
        s = cma._init_state(5, torch.device("cpu"))
        assert s['p_sigma'].shape == (5,)
        assert s['p_c'].shape == (5,)
        assert s['C'].shape == (5, 5)
        assert s['BD'].shape == (5, 5)


class TestCMAESStep:

    def test_single_step(self):
        """One generation should run without error."""
        cma = CMAES(sigma=0.5, pop_size=6)
        s = cma._init_state(3, torch.device("cpu"))
        s['mean'] = torch.zeros(3, dtype=torch.float64)

        def eval_fn(pop):
            return (pop ** 2).sum(dim=1)

        cost, best = cma._step(s, eval_fn)
        assert isinstance(cost, float)
        assert best.shape == (3,)

    def test_mean_updates(self):
        """Mean should change after a step."""
        cma = CMAES(sigma=1.0, pop_size=10)
        s = cma._init_state(3, torch.device("cpu"))
        s['mean'] = torch.zeros(3, dtype=torch.float64)
        mean_before = s['mean'].clone()

        def eval_fn(pop):
            return (pop ** 2).sum(dim=1)

        cma._step(s, eval_fn)
        # Mean should have moved (with overwhelming probability)
        assert not torch.allclose(s['mean'], mean_before)

    def test_sigma_adapts(self):
        """Sigma should change after a step."""
        cma = CMAES(sigma=1.0, pop_size=10)
        s = cma._init_state(5, torch.device("cpu"))
        s['mean'] = torch.zeros(5, dtype=torch.float64)
        sigma_before = s['sigma']

        def eval_fn(pop):
            return (pop ** 2).sum(dim=1)

        cma._step(s, eval_fn)
        assert s['sigma'] != sigma_before


class TestCMAESConvergence:

    def test_sphere_function(self):
        """CMA-ES should minimize f(x) = sum(x^2) close to 0."""
        cma = CMAES(sigma=1.0, pop_size=10)
        n = 3
        s = cma._init_state(n, torch.device("cpu"))
        s['mean'] = torch.randn(n, dtype=torch.float64) * 2

        def eval_fn(pop):
            return (pop ** 2).sum(dim=1)

        for _ in range(50):
            cost, _ = cma._step(s, eval_fn)

        # Should be close to 0
        assert cost < 0.1

    def test_shifted_sphere(self):
        """CMA-ES should find minimum of f(x) = sum((x-1)^2)."""
        cma = CMAES(sigma=1.0, pop_size=10)
        n = 3
        s = cma._init_state(n, torch.device("cpu"))
        s['mean'] = torch.zeros(n, dtype=torch.float64)

        def eval_fn(pop):
            return ((pop - 1.0) ** 2).sum(dim=1)

        for _ in range(100):
            cost, best = cma._step(s, eval_fn)

        assert cost < 0.1
        assert torch.allclose(best, torch.ones(n, dtype=torch.float64), atol=0.3)


class TestMinimizeCMAES:

    def test_ry_z_optimization(self):
        """Optimize RY(theta) to minimize <Z>. Minimum at theta=pi (cos(pi)=-1)."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")

        cma = CMAES(sigma=0.5, pop_size=8)
        theta0 = torch.tensor([0.5])
        theta, exp_values = minimize_cma_es(model, theta0, cma, generations=30)

        # Should converge toward theta=pi, E=-1
        assert exp_values[-1] < -0.8

    def test_returns_correct_shapes(self):
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")

        cma = CMAES(sigma=0.5, pop_size=6)
        theta0 = torch.tensor([0.0])
        theta, exp_values = minimize_cma_es(model, theta0, cma, generations=5)

        assert theta.shape == (1,)
        assert len(exp_values) == 5

    def test_2param_optimization(self):
        """2-param circuit should converge."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        op = SparsePauliOp.from_list([("ZI", 0.5), ("IZ", 0.5)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")

        cma = CMAES(sigma=0.5, pop_size=10)
        theta0 = torch.tensor([0.0, 0.0])
        theta, exp_values = minimize_cma_es(model, theta0, cma, generations=30)

        # Minimum: both thetas at pi, E = 0.5*(-1) + 0.5*(-1) = -1
        assert exp_values[-1] < -0.5
