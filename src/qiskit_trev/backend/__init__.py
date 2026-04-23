"""Backend abstraction for qiskit-trev tensor-ring kernels.

Scope (plan/14 Step 1–2): narrow protocol covering only the ops reachable
from ``measure.efficient_contraction.batched_expectation_value``. Torch
and JAX implementations are available; the JAX module is imported lazily
so torch-only installs do not require JAX.
"""

from ._protocol import Backend
from ._torch import TORCH_BACKEND, TorchBackend


def get_backend(x=None) -> Backend:
    """Return the backend matching the input array type.

    - ``torch.Tensor`` or ``None`` → :data:`TORCH_BACKEND`.
    - ``jax.Array`` / ``jaxlib.Array`` → :data:`JAX_BACKEND` (JAX imported lazily).
    """
    if x is not None:
        mod = type(x).__module__
        if mod.startswith("jax") or mod.startswith("jaxlib"):
            from ._jax import JAX_BACKEND
            return JAX_BACKEND
    return TORCH_BACKEND


_LAZY_JAX_ATTRS = ("JAX_BACKEND", "JAX_BACKEND_HIGHEST", "JaxBackend")


def __getattr__(name: str):
    # Lazy access to the JAX symbols so torch-only installs do not require
    # jax at import time.
    if name in _LAZY_JAX_ATTRS:
        from . import _jax

        return getattr(_jax, name)
    raise AttributeError(f"module 'qiskit_trev.backend' has no attribute {name!r}")


__all__ = [
    "Backend",
    "TorchBackend",
    "TORCH_BACKEND",
    "JaxBackend",
    "JAX_BACKEND",
    "JAX_BACKEND_HIGHEST",
    "get_backend",
]
