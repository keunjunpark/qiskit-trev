"""Cache-key normalisation: equivalent floats from different sources
should hash to the same key."""

from __future__ import annotations

import math

import numpy as np

from qiskit_trev.backend._cache_utils import canonical_shift


def test_python_float_and_numpy_float64_collide():
    a = canonical_shift(math.pi / 2)
    b = canonical_shift(np.float64(math.pi / 2))
    assert a == b
    assert {a: 1, b: 2} == {a: 2}


def test_arithmetic_drift_collapses():
    # 0.1 + 0.2 != 0.3 in fp; canonical_shift should still fold them.
    assert canonical_shift(0.1 + 0.2) == canonical_shift(0.3)


def test_returns_python_float():
    out = canonical_shift(np.float64(1.234567890123456))
    assert isinstance(out, float)


def test_distinct_shifts_remain_distinct():
    # Sanity: two shifts that differ by more than 1e-12 must remain
    # different keys.
    assert canonical_shift(0.1) != canonical_shift(0.2)
    assert canonical_shift(math.pi / 2) != canonical_shift(math.pi / 4)
