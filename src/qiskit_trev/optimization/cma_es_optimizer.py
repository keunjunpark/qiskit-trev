"""CMA-ES optimizer with unified interface and batched evaluation."""

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
        eval_batch_size: Batch size for evaluating population members.
            None = evaluate all at once. Set smaller to control GPU memory.
    """

    def __init__(
        self,
        sigma: float = 0.5,
        pop_size: int | None = None,
        eigen_every: int = 1,
        eval_batch_size: int | None = None,
    ):
        self._cma = CMAES(sigma=sigma, pop_size=pop_size, eigen_every=eigen_every)
        self._eval_batch_size = eval_batch_size

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
        eval_bs = self._eval_batch_size

        def evaluate_fn(population: Tensor) -> Tensor:
            return model.evaluate_batch(population, batch_size=eval_bs).float()

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
