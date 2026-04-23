"""Backend protocol — narrow surface for Step 1."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """Array-library abstraction used by batched kernel code paths.

    Only the ops reachable from ``batched_expectation_value`` live here. As
    more kernels are ported behind the backend (plan/14 Step 3+), ops are
    added to this protocol.
    """

    name: str
    complex_dtype: Any

    def zeros(self, shape: Sequence[int] | int, *, device: Any = None) -> Any: ...

    def eye(self, n: int, *, device: Any = None) -> Any: ...

    def tensor(self, data: Any, *, device: Any = None) -> Any: ...

    def as_tensor(self, data: Any, *, device: Any = None) -> Any: ...

    def einsum(self, spec: str, *operands: Any) -> Any: ...

    def where(self, cond: Any, x: Any, y: Any) -> Any: ...

    def unsqueeze(self, x: Any, dim: int) -> Any: ...

    def reshape(self, x: Any, shape: Sequence[int]) -> Any: ...

    def to_device(self, x: Any, device: Any) -> Any: ...
