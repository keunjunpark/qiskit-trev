"""Backend abstraction for qiskit-trev tensor-ring kernels.

Scope (plan/14 Step 1–2): narrow protocol covering only the ops reachable
from ``measure.efficient_contraction.batched_expectation_value``. Torch
and JAX implementations are available; the JAX module is imported lazily
so torch-only installs do not require JAX.
"""

import os

from ._protocol import Backend
from ._torch import TORCH_BACKEND, TorchBackend


_VALID_AUTOTUNE_LEVELS = (0, 2, 4)
_CACHE_DIR_ENV = "QISKIT_TREV_CACHE_DIR"


def _resolve_cache_dir(cache_dir: str | None) -> str:
    """Pick the cache directory, in priority order:
    explicit arg → ``QISKIT_TREV_CACHE_DIR`` env var → ``~/.cache/qiskit_trev_jax``.
    """
    if cache_dir is not None:
        return cache_dir
    env_dir = os.environ.get(_CACHE_DIR_ENV)
    if env_dir:
        return env_dir
    return os.path.expanduser("~/.cache/qiskit_trev_jax")


def _apply_xla_flag(name: str, value: str) -> None:
    """Add or replace ``--<name>=<value>`` in the ``XLA_FLAGS`` env var
    without dropping any pre-existing flags the user has set."""
    cur = os.environ.get("XLA_FLAGS", "")
    parts = [p for p in cur.split() if not p.startswith(f"--{name}=")]
    parts.append(f"--{name}={value}")
    os.environ["XLA_FLAGS"] = " ".join(parts).strip()


def enable_compilation_cache(
    cache_dir: str | None = None,
    *,
    autotune_level: int = 2,
) -> str:
    """Enable JAX's on-disk compilation cache.

    First call in a given Python process still pays the full XLA compile
    (1–10 s on GPU, depending on parameter count). With the cache
    enabled, *subsequent Python processes* load the compiled artifact
    from disk in ~0.3 s instead of recompiling — this is what makes
    notebook restarts and short scripts bearable.

    Opt-in rather than auto-on because it writes to the filesystem and
    users should control where.

    Args:
        cache_dir: Directory to persist compiled artifacts. Resolution
            order: explicit arg → ``QISKIT_TREV_CACHE_DIR`` env var →
            ``~/.cache/qiskit_trev_jax``. Created if missing. The env
            var lets you point the cache at platform-specific persistent
            storage (Google Drive on Colab, ``$SCRATCH`` on HPC,
            ``/kaggle/working`` on Kaggle, etc.) without library code
            knowing about any of those.
        autotune_level: XLA's GEMM autotune aggressiveness. ``2``
            (default) is the recommended balance — well-tuned kernels
            without paying for exhaustive search. ``0`` disables
            autotune for the fastest compile at ~10–20 % steady-state
            cost. ``4`` is the XLA default — safest kernels but slowest
            compile. Must be one of ``{0, 2, 4}``. Set before any
            ``jax.jit`` call so XLA picks it up at first compile.

    Returns:
        The resolved cache directory path.

    Examples:
        Local default::

            from qiskit_trev.backend import enable_compilation_cache
            enable_compilation_cache()  # ~/.cache/qiskit_trev_jax

        Colab with Drive (set the env var before calling)::

            import os
            os.environ["QISKIT_TREV_CACHE_DIR"] = (
                "/content/drive/MyDrive/qiskit_trev_jax_cache"
            )
            enable_compilation_cache()

        Faster compile, slower kernels (rapid prototyping)::

            enable_compilation_cache(autotune_level=0)
    """
    import jax

    if autotune_level not in _VALID_AUTOTUNE_LEVELS:
        raise ValueError(
            f"autotune_level must be one of {_VALID_AUTOTUNE_LEVELS}, "
            f"got {autotune_level!r}"
        )

    cache_dir = _resolve_cache_dir(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    _apply_xla_flag("xla_gpu_autotune_level", str(int(autotune_level)))
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
