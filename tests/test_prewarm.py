"""prewarm should populate the JIT cache so subsequent calls don't
recompile."""

from __future__ import annotations

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from qiskit_trev.gradient import BatchParameterShiftGradient
from qiskit_trev.model import TensorRingModel
from qiskit_trev.prewarm import prewarm


def _tiny_model():
    """Smallest plausible workload — keeps the test fast."""
    qc = QuantumCircuit(3)
    for q in range(3):
        qc.ry(0.0, q)
        qc.cx(q, (q + 1) % 3)
    obs = SparsePauliOp.from_list([("ZII", 1.0), ("IZI", 0.5)])
    return TensorRingModel(qc, obs, rank=4), 3


def test_prewarm_populates_cache():
    model, P = _tiny_model()
    grad_fn = BatchParameterShiftGradient(model, backend="jax")
    assert len(grad_fn._jax_jit_cache) == 0

    prewarm(grad_fn, [P])
    assert len(grad_fn._jax_jit_cache) > 0

    # Second prewarm with the same P must not add new entries.
    keys_before = set(grad_fn._jax_jit_cache.keys())
    prewarm(grad_fn, [P])
    assert set(grad_fn._jax_jit_cache.keys()) == keys_before


def test_prewarm_skips_duplicates_and_non_positive():
    model, P = _tiny_model()
    grad_fn = BatchParameterShiftGradient(model, backend="jax")
    # Mix duplicates and a 0 / negative; only P should compile.
    prewarm(grad_fn, [P, P, 0, -1, P])
    assert len(grad_fn._jax_jit_cache) > 0


def test_grad_call_after_prewarm_hits_cache():
    model, P = _tiny_model()
    grad_fn = BatchParameterShiftGradient(model, backend="jax")
    prewarm(grad_fn, [P])
    keys_after_warm = set(grad_fn._jax_jit_cache.keys())

    # A real grad call with the same P must reuse, not recompile.
    grad_fn(jnp.zeros(P, dtype=jnp.float32)).block_until_ready()
    assert set(grad_fn._jax_jit_cache.keys()) == keys_after_warm
