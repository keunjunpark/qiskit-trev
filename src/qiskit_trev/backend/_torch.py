"""PyTorch implementation of :class:`Backend`."""

from __future__ import annotations

from typing import Any, Sequence

import torch


class TorchBackend:
    name = "torch"
    complex_dtype = torch.cfloat

    def zeros(self, shape: Sequence[int] | int, *, device: Any = None):
        return torch.zeros(shape, dtype=self.complex_dtype, device=device)

    def eye(self, n: int, *, device: Any = None):
        return torch.eye(n, dtype=self.complex_dtype, device=device)

    def tensor(self, data: Any, *, device: Any = None):
        return torch.tensor(data, dtype=self.complex_dtype, device=device)

    def as_tensor(self, data: Any, *, device: Any = None):
        return torch.as_tensor(data, dtype=self.complex_dtype, device=device)

    def einsum(self, spec: str, *operands: Any):
        return torch.einsum(spec, *operands)

    def where(self, cond: Any, x: Any, y: Any):
        return torch.where(cond, x, y)

    def unsqueeze(self, x: Any, dim: int):
        return x.unsqueeze(dim)

    def reshape(self, x: Any, shape: Sequence[int]):
        return x.reshape(shape)

    def to_device(self, x: Any, device: Any):
        return x.to(device)


TORCH_BACKEND = TorchBackend()
