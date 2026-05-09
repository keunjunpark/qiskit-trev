"""Pre-compile the JAX gradient kernel for known shapes.

The first call to a JAX-backed ``BatchParameterShiftGradient`` for a
given parameter count ``P`` triggers an XLA compile that scales with
``P`` (~95 ms / parameter on a current-gen GPU). A 100-parameter
ansatz costs ~10 s up-front.

``prewarm`` runs that compile in a controlled place — typically right
after constructing your gradient function and before training — so the
user pays the cost once, in a phase where they expect setup latency,
instead of discovering it on the first training-loop step.

When :func:`qiskit_trev.backend.enable_compilation_cache` is also
active, the compile artefact additionally lands on disk and is reused
across process restarts.
"""

from __future__ import annotations

from typing import Iterable


def prewarm(grad_fn, P_list: Iterable[int]) -> None:
    """Warm ``grad_fn``'s JAX JIT cache for each parameter count in
    ``P_list``.

    For every unique ``P``, calls ``grad_fn`` once with a zero vector of
    length ``P``. That populates ``grad_fn``'s in-process cache so the
    next real call with a same-shape input is sub-ms. Pair with
    :func:`qiskit_trev.backend.enable_compilation_cache` to make the
    compile survive process restarts.

    Args:
        grad_fn: A callable that accepts a 1-D float array. Designed
            for :class:`~qiskit_trev.gradient.BatchParameterShiftGradient`
            constructed with ``backend="jax"``, but works with any
            JAX-backed callable that takes a 1-D parameter vector.
        P_list: Iterable of parameter counts. Duplicates and
            non-positive values are skipped.

    Example::

        from qiskit_trev.backend import enable_compilation_cache
        from qiskit_trev.gradient import BatchParameterShiftGradient
        from qiskit_trev.prewarm import prewarm

        enable_compilation_cache()
        grad_fn = BatchParameterShiftGradient(model, backend="jax")
        prewarm(grad_fn, P_list=[16, 32, 64])
        # Training loop now starts immediately — no compile pause.
    """
    import numpy as np
    import jax.numpy as jnp

    seen: set[int] = set()
    for P in P_list:
        P = int(P)
        if P <= 0 or P in seen:
            continue
        seen.add(P)
        params = jnp.asarray(np.zeros(P, dtype=np.float32))
        result = grad_fn(params)
        result.block_until_ready()
