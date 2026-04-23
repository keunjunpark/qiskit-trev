"""JAX implementation of :class:`Backend`.

Lands with plan/14 Step 2. Imports live at module level — importing this
module implies JAX is installed. The package ``__init__`` defers the import
so torch-only installs keep working.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp


def _to_numpy(data: Any) -> Any:
    """Best-effort conversion from torch/array-like to something ``jnp.asarray`` accepts."""
    if hasattr(data, "detach") and hasattr(data, "cpu"):  # torch.Tensor duck-type
        return data.detach().cpu().numpy()
    return data


class JaxBackend:
    """JAX backend.

    ``precision`` controls XLA matmul precision for ``einsum`` —
    ``Precision.DEFAULT`` on GPU uses tf32-like reduced-mantissa
    accumulation (fast but ~1e-4 relative error vs torch), ``HIGHEST``
    uses full fp32 (bit-identical to torch, slower). Parity tests should
    use ``HIGHEST``; benchmarks use ``DEFAULT`` because that is what
    users get in practice.
    """

    name = "jax"
    complex_dtype = jnp.complex64

    def __init__(self, precision: jax.lax.Precision = jax.lax.Precision.DEFAULT):
        self.precision = precision

    def zeros(self, shape: Sequence[int] | int, *, device: Any = None):
        arr = jnp.zeros(shape, dtype=self.complex_dtype)
        return arr if device is None else jax.device_put(arr, device)

    def eye(self, n: int, *, device: Any = None):
        arr = jnp.eye(n, dtype=self.complex_dtype)
        return arr if device is None else jax.device_put(arr, device)

    def tensor(self, data: Any, *, device: Any = None):
        arr = jnp.asarray(_to_numpy(data), dtype=self.complex_dtype)
        return arr if device is None else jax.device_put(arr, device)

    def as_tensor(self, data: Any, *, device: Any = None):
        arr = jnp.asarray(_to_numpy(data), dtype=self.complex_dtype)
        return arr if device is None else jax.device_put(arr, device)

    def einsum(self, spec: str, *operands: Any):
        return jnp.einsum(spec, *operands, precision=self.precision)

    def where(self, cond: Any, x: Any, y: Any):
        return jnp.where(cond, x, y)

    def unsqueeze(self, x: Any, dim: int):
        return jnp.expand_dims(x, axis=dim)

    def reshape(self, x: Any, shape: Sequence[int]):
        return x.reshape(shape)

    def to_device(self, x: Any, device: Any):
        arr = jnp.asarray(_to_numpy(x))
        return arr if device is None else jax.device_put(arr, device)


JAX_BACKEND = JaxBackend()
JAX_BACKEND_HIGHEST = JaxBackend(precision=jax.lax.Precision.HIGHEST)
