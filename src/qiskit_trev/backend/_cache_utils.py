"""Utilities for normalising JIT-cache keys.

The in-memory ``_jax_jit_cache`` dicts in ``gradient.py`` and ``qml.py``
key on Python tuples that include the parameter-shift ``shift`` value.
Floats compare by exact bit pattern, so ``math.pi / 2`` (Python float)
and ``torch.pi / 2`` (numpy.float64) produce different keys despite
being numerically identical — every such miss costs a fresh ``jax.jit``
compile (1–10 s on GPU).

``canonical_shift`` rounds to 12 decimal digits, which collapses
bit-level noise without losing meaningful precision for any shift a
user would realistically choose.
"""

from __future__ import annotations


_SHIFT_DECIMALS = 12


def canonical_shift(shift) -> float:
    """Normalise a parameter-shift value for use as a dict key.

    Accepts any real-valued scalar (Python float, numpy scalar, 0-d
    tensor, etc.) and returns a Python float rounded to a fixed number
    of decimal digits, so equivalent values from different sources hit
    the same JIT-cache slot.
    """
    return round(float(shift), _SHIFT_DECIMALS)
