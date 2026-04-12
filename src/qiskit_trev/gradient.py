"""Batched parameter-shift gradient computation.

Constructs shifted parameter batches and evaluates them using batched
tensor ring building and measurement for efficient gradient computation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .model import TensorRingModel
from .tensor_ring.state import TensorRingState
from .measure.efficient_contraction import expectation_value as ev_efficient
from .measure.full_contraction import expectation_value as ev_full


class BatchParameterShiftGradient:
    """Compute parameter-shift gradients using batched tensor ring evaluation.

    Constructs all shifted parameter vectors at once and evaluates them
    in chunks using batched state building, avoiding redundant computation.

    Args:
        model: TensorRingModel to compute gradients for.
        shift: Parameter shift amount (default pi/2).
        chunk_size: Number of parameter shifts to evaluate per chunk.
            Larger chunks use more memory but are faster on GPU.
    """

    def __init__(
        self,
        model: TensorRingModel,
        shift: float = math.pi / 2,
        chunk_size: int | None = None,
    ):
        self._model = model
        self._shift = shift
        self._chunk_size = chunk_size

    @torch.no_grad()
    def __call__(self, params: Tensor) -> Tensor:
        """Compute gradient via batched parameter shift.

        Args:
            params: (P,) tensor of parameter values.

        Returns:
            (P,) tensor of gradients.
        """
        P = len(params)
        if P == 0:
            return torch.zeros(0, dtype=torch.float64)

        model = self._model
        shift = self._shift
        denom = 2 * math.sin(shift)
        chunk_size = self._chunk_size or P

        grad = torch.zeros(P, dtype=torch.float64)

        for start in range(0, P, chunk_size):
            stop = min(start + chunk_size, P)
            C = stop - start

            # Build (2C, P) batch: first C are +shift, last C are -shift
            base = params.unsqueeze(0).expand(2 * C, -1).clone()
            idx = torch.arange(start, stop)
            arange_C = torch.arange(C)

            base[arange_C, idx] += shift
            base[C + arange_C, idx] -= shift

            # Build all tensor rings in batch
            state = TensorRingState(
                model._num_qubits, model.rank, model.device_str, model.dtype
            )
            batch_tensor = state.build_batch(model._gate_templates, base)

            # Evaluate expectation values
            evs = torch.zeros(2 * C, dtype=torch.float64)
            for i in range(2 * C):
                if model._use_efficient:
                    evs[i] = ev_efficient(batch_tensor[i], model._hamiltonian)
                else:
                    evs[i] = ev_full(batch_tensor[i], model._hamiltonian)

            # Gradient: (E+ - E-) / (2*sin(shift))
            grad[start:stop] = (evs[:C] - evs[C:]) / denom

        return grad
