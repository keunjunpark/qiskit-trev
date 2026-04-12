"""Base optimizer interface for qiskit-trev."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..model import TensorRingModel


@dataclass
class OptimizationResult:
    """Result of an optimization run.

    Attributes:
        params: Optimized parameter vector.
        cost: Final cost (expectation value).
        cost_history: Cost at each iteration.
        num_iterations: Number of iterations completed.
    """
    params: Tensor
    cost: float
    cost_history: list[float] = field(default_factory=list)
    num_iterations: int = 0


class Optimizer(ABC):
    """Base optimizer interface.

    All optimizers implement `minimize(model, theta0, max_iter)` and
    return an `OptimizationResult`.
    """

    @abstractmethod
    def minimize(
        self,
        model: TensorRingModel,
        theta0: Tensor,
        max_iter: int = 100,
    ) -> OptimizationResult:
        """Run optimization.

        Args:
            model: TensorRingModel to optimize.
            theta0: (P,) initial parameter vector.
            max_iter: Maximum number of iterations.

        Returns:
            OptimizationResult with optimized parameters and history.
        """
