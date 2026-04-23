"""Backend abstraction for qiskit-trev tensor-ring kernels.

Scope (plan/14 Step 1–2): narrow protocol covering only the ops reachable
from ``measure.efficient_contraction.batched_expectation_value``. Torch
and JAX implementations are available; the JAX module is imported lazily
so torch-only installs do not require JAX.
"""

import os

from ._protocol import Backend
from ._torch import TORCH_BACKEND, TorchBackend


def enable_compilation_cache(cache_dir: str | None = None) -> str:
    """Enable JAX's on-disk compilation cache.

    First call in a given Python process still pays the full XLA compile
    (1–3 s on GPU, 30–50 s on TPU). With the cache enabled, *subsequent
    Python processes* load the compiled artifact from disk (sub-second)
    instead of recompiling — this is what makes notebook restarts and
    short scripts bearable.

    Opt-in rather than auto-on because it writes to the filesystem and
    users should control where.

    Args:
        cache_dir: Directory to persist compiled artifacts. Defaults to
            ``~/.cache/qiskit_trev_jax``. Created if missing.

    Returns:
        The resolved cache directory path.

    Example:
        >>> from qiskit_trev.backend import enable_compilation_cache
        >>> enable_compilation_cache()  # ~/.cache/qiskit_trev_jax
    """
    import jax

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/qiskit_trev_jax")
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    return cache_dir


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
    "enable_compilation_cache",
    "get_backend",
]
