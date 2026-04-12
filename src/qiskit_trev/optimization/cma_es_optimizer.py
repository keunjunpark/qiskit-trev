"""CMA-ES optimizer with unified interface."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Optimizer, OptimizationResult
from .cma_es import CMAES
from ..model import TensorRingModel


class CMAESOptimizer(Optimizer):
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.

    Derivative-free, population-based optimizer that adapts a multivariate
    Gaussian to learn parameter correlations.

    Args:
        sigma: Initial step size. Typical: 0.5-1.0.
        pop_size: Population size per generation. None = auto.
        eigen_every: Eigendecomposition frequency.
    """

    def __init__(
        self,
        sigma: float = 0.5,
        pop_size: int | None = None,
        eigen_every: int = 1,
    ):
        self._cma = CMAES(sigma=sigma, pop_size=pop_size, eigen_every=eigen_every)

    @torch.no_grad()
    def minimize(
        self,
        model: TensorRingModel,
        theta0: Tensor,
        max_iter: int = 100,
    ) -> OptimizationResult:
        device = torch.device(model.device_str)
        n = theta0.shape[0]

        s = self._cma._init_state(n, device)
        s['mean'] = theta0.double().to(device)

        cost_history: list[float] = []

        def evaluate_fn(population: Tensor) -> Tensor:
            lam = population.shape[0]
            costs = torch.zeros(lam, dtype=torch.float32)
            for i in range(lam):
                costs[i] = model(population[i]).item()
            return costs

        best_cost = float('inf')
        best_params = theta0.clone()

        for _ in range(max_iter):
            cost, params = self._cma._step(s, evaluate_fn)
            cost_history.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_params = params.float()

        return OptimizationResult(
            params=s['mean'].float(),
            cost=best_cost,
            cost_history=cost_history,
            num_iterations=max_iter,
        )
