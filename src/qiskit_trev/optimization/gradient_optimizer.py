"""Gradient-based optimizer with unified interface."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import Optimizer, OptimizationResult
from ..model import TensorRingModel
from ..gradient import BatchParameterShiftGradient


class GradientOptimizer(Optimizer):
    """Gradient-based optimizer using parameter-shift rule.

    Uses batched parameter-shift for gradient computation and a
    PyTorch optimizer (Adam or SGD) for parameter updates.

    Args:
        lr: Learning rate.
        optimizer_cls: "adam" or "sgd". Default "sgd".
        shift: Parameter shift amount for gradient computation.
        chunk_size: Batch chunk size for gradient computation.
    """

    def __init__(
        self,
        lr: float = 0.01,
        optimizer_cls: str = "sgd",
        shift: float = math.pi / 2,
        chunk_size: int | None = None,
    ):
        self._lr = lr
        self._optimizer_cls = optimizer_cls.lower()
        self._shift = shift
        self._chunk_size = chunk_size

    @torch.no_grad()
    def minimize(
        self,
        model: TensorRingModel,
        theta0: Tensor,
        max_iter: int = 100,
    ) -> OptimizationResult:
        params = theta0.clone().float()
        grad_fn = BatchParameterShiftGradient(
            model, shift=self._shift, chunk_size=self._chunk_size
        )

        # Create a dummy parameter for the torch optimizer
        p = torch.nn.Parameter(params.clone())
        if self._optimizer_cls == "adam":
            opt = torch.optim.Adam([p], lr=self._lr)
        else:
            opt = torch.optim.SGD([p], lr=self._lr)

        cost_history: list[float] = []

        for _ in range(max_iter):
            opt.zero_grad()
            cost = model(p.data).item()
            cost_history.append(cost)
            grad = grad_fn(p.data)
            p.grad = grad.float()
            opt.step()

        final_cost = model(p.data).item()

        return OptimizationResult(
            params=p.data.detach(),
            cost=final_cost,
            cost_history=cost_history,
            num_iterations=max_iter,
        )
