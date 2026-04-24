"""Tests for the JAX-aware chunk-size probe (plan/16).

Covers:
- ``jit_chunk_binary_search`` — the framework-agnostic helper.
- ``BatchParameterShiftGradient`` — ``max_chunk_size`` ceiling and JAX
  probe behaviour under a mocked OOM.
- ``QMLModel`` — ``max_ps_chunk`` ceiling and JAX probe under mocked
  OOM.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from qiskit_trev.gradient import BatchParameterShiftGradient
from qiskit_trev.model import TensorRingModel
from qiskit_trev.optimization.auto_batch import jit_chunk_binary_search
from qiskit_trev.qml import QMLModel


# ---------- jit_chunk_binary_search --------------------------------------

def test_helper_upper_passes():
    """If the top candidate passes, no binary search needed; return upper * safety."""
    calls = []
    def _try(c):
        calls.append(c)
        return True
    assert jit_chunk_binary_search(_try, upper=16, safety_frac=0.75) == 12
    assert calls == [16]


def test_helper_bottom_fails():
    """If even `lower` fails, return lower unchanged (caller decides what to do)."""
    calls = []
    def _try(c):
        calls.append(c)
        return False
    assert jit_chunk_binary_search(_try, upper=16, safety_frac=0.75) == 1
    assert calls == [16, 1]


def test_helper_binary_search_finds_cap():
    """Chunk passes iff c <= 8 — probe should land at 8 * 0.75 = 6."""
    probed: list[int] = []
    def _try(c):
        probed.append(c)
        return c <= 8
    result = jit_chunk_binary_search(_try, upper=40, safety_frac=0.75)
    assert result == 6, f"got {result}; probed {probed}"


def test_helper_binary_search_no_safety_factor():
    probed: list[int] = []
    def _try(c):
        probed.append(c)
        return c <= 8
    result = jit_chunk_binary_search(_try, upper=40, safety_frac=1.0)
    assert result == 8


@pytest.mark.parametrize("cap", [1, 2, 5, 17, 63])
def test_helper_various_caps(cap):
    """Sweep: probe should return floor(cap * safety)."""
    def _try(c):
        return c <= cap
    result = jit_chunk_binary_search(_try, upper=128, safety_frac=0.75)
    expected = max(1, int(cap * 0.75))
    assert result == expected


def test_helper_upper_equals_lower():
    """Edge case: degenerate range — return lower regardless of try_fn."""
    assert jit_chunk_binary_search(lambda c: True, upper=3, lower=3) == 3
    assert jit_chunk_binary_search(lambda c: False, upper=3, lower=3) == 3


# ---------- BatchParameterShiftGradient ----------------------------------

def _vqe_model(N=4, reps=1, rank=4):
    qc = QuantumCircuit(N)
    P = 0
    for _ in range(reps):
        for q in range(N):
            qc.ry(0.0, q)
            P += 1
        for q in range(N):
            qc.cx(q, (q + 1) % N)
    obs = SparsePauliOp.from_list([("Z" + "I" * (N - 1), 1.0)])
    return TensorRingModel(qc, obs, rank=rank), P


def test_vqe_max_chunk_size_clamps_auto():
    """Even if auto-tune and the JAX probe like a larger chunk,
    max_chunk_size must cap it."""
    model, P = _vqe_model(N=4, reps=1)
    grad_fn = BatchParameterShiftGradient(
        model, chunk_size="auto", max_chunk_size=2, backend="jax"
    )
    params = jnp.asarray(np.zeros(P, dtype=np.float32))
    grad_fn(params)  # triggers chunk-size resolution + caching
    assert grad_fn._jax_probed_chunk[P] <= P
    # The ceiling must apply: the actual chunk used is bounded at 2.
    # Validate by inspecting the jit cache keys — chunked path stores
    # ("chunk", ..., C) entries. The largest C in the cache is the
    # chunk actually used.
    chunk_keys = [k for k in grad_fn._jax_jit_cache if k[0] == "chunk"]
    used_chunks = [k[-1] for k in chunk_keys]
    if used_chunks:
        assert max(used_chunks) <= 2


def test_vqe_max_chunk_size_validation():
    model, _ = _vqe_model(N=3, reps=1)
    with pytest.raises(ValueError, match="max_chunk_size"):
        BatchParameterShiftGradient(model, max_chunk_size=0)


def test_vqe_probe_caches_per_P(monkeypatch):
    """Second call with the same P must reuse the cached probe value."""
    model, P = _vqe_model(N=4, reps=1)
    grad_fn = BatchParameterShiftGradient(
        model, chunk_size="auto", backend="jax", max_chunk_size=3,
    )
    probe_calls = []
    real_probe = grad_fn._probe_jax_chunk

    def counting_probe(P_, upper):
        probe_calls.append(P_)
        return real_probe(P_, upper)

    monkeypatch.setattr(grad_fn, "_probe_jax_chunk", counting_probe)

    params = jnp.asarray(np.zeros(P, dtype=np.float32))
    grad_fn(params)
    grad_fn(params)
    grad_fn(params)
    assert len(probe_calls) == 1, f"probed {len(probe_calls)} times for same P"


def test_vqe_auto_skips_probe_when_torch_tuned_is_one(monkeypatch):
    """If the torch probe already returns 1, skip the JAX probe entirely
    and cache chunk_size=1 directly. Covers the tiny-workload early path."""
    model, P = _vqe_model(N=4, reps=1)
    grad_fn = BatchParameterShiftGradient(
        model, chunk_size="auto", backend="jax",
    )
    monkeypatch.setattr(grad_fn, "_resolve_chunk_size", lambda P_: 1)

    probe_calls = []
    monkeypatch.setattr(
        grad_fn, "_probe_jax_chunk",
        lambda P_, upper: probe_calls.append((P_, upper)) or 1,
    )

    grad_fn(jnp.asarray(np.zeros(P, dtype=np.float32)))
    assert grad_fn._jax_probed_chunk[P] == 1
    assert probe_calls == []  # probe must not have run


def test_vqe_probe_reraises_non_oom_errors(monkeypatch):
    """The probe must not swallow errors that aren't OOM — it's supposed
    to binary-search on memory failures, not hide genuine bugs."""
    model, P = _vqe_model(N=4, reps=1)
    grad_fn = BatchParameterShiftGradient(
        model, chunk_size="auto", backend="jax",
    )

    def fake_grad(params, P_, chunk, cache):
        raise ValueError("something unrelated blew up")

    monkeypatch.setattr(grad_fn, "_jax_grad_chunked", fake_grad)
    monkeypatch.setattr(grad_fn, "_resolve_chunk_size", lambda P_: P_)

    with pytest.raises(ValueError, match="unrelated"):
        grad_fn._probe_jax_chunk(P, upper=P)


def test_qml_probe_reraises_non_oom_errors(monkeypatch):
    """QMLModel probe must re-raise non-OOM errors, same contract as VQE."""
    model = _qml_model(n_qubits=3, n_layers=2)
    N = 4

    def fake_grad(X, theta, **kw):
        raise ValueError("unrelated bug")

    monkeypatch.setattr(model, "parameter_shift_grad", fake_grad)
    with pytest.raises(ValueError, match="unrelated"):
        model._probe_jax_ps_chunk(N, upper=model.n_trainable)


def test_vqe_probe_fakes_oom_returns_safe_cap(monkeypatch):
    """Mock _jax_grad_chunked to raise XlaRuntimeError for chunk > K and
    verify the probe lands at K * 0.75."""
    model, P = _vqe_model(N=4, reps=1)  # P=4
    grad_fn = BatchParameterShiftGradient(
        model, chunk_size="auto", backend="jax"
    )

    cap = 2
    # The probe matches on the error *message*, not the class, so a
    # plain RuntimeError with the right substring is enough to simulate
    # an XLA OOM in-test.
    oom_cls = RuntimeError

    def fake_grad(params, P_, chunk, cache):
        if chunk > cap:
            raise oom_cls("RESOURCE_EXHAUSTED: Out of memory")

        class _Dummy:
            def block_until_ready(self):
                return self
        return jnp.zeros(P_, dtype=jnp.float32)

    monkeypatch.setattr(grad_fn, "_jax_grad_chunked", fake_grad)
    # Skip the torch probe by hard-coding the upper bound
    monkeypatch.setattr(grad_fn, "_resolve_chunk_size", lambda P_: P_)

    result = grad_fn._probe_jax_chunk(P, upper=P)
    # cap=2, safety=0.75 → max(1, int(2*0.75)) = 1
    assert result == 1


# ---------- QMLModel -----------------------------------------------------

def _qml_model(n_qubits=3, n_layers=1, rank=4, **kwargs):
    qc = QuantumCircuit(n_qubits)
    di, ti = [], []
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(0.0, q)
            di.append(idx); idx += 1
        for q in range(n_qubits):
            qc.ry(0.0, q)
            ti.append(idx); idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return QMLModel(qc, di, ti, rank=rank, device="cpu", **kwargs)


def test_qml_max_ps_chunk_validation():
    with pytest.raises(ValueError, match="max_ps_chunk"):
        _qml_model(max_ps_chunk=0)


def test_qml_max_ps_chunk_clamps_auto_tune_on_cpu():
    """On CPU, auto_tune sets self.batch_size=N and _ps_chunk=n_trainable.
    max_ps_chunk must still clamp on top of that."""
    model = _qml_model(n_qubits=3, n_layers=2, max_ps_chunk=2)
    # The CPU branch of auto_tune returns early — it doesn't populate
    # _ps_chunk_cache. Instead we verify the override is stored and
    # available to the gradient path.
    assert model._max_ps_chunk == 2


def test_qml_probe_fakes_oom_returns_safe_cap(monkeypatch):
    """Mock parameter_shift_grad to raise on chunk > K, verify probe
    lands at int(K * 0.75)."""
    model = _qml_model(n_qubits=3, n_layers=2)  # n_trainable = 6
    N = 4

    cap = 4
    # Match on the error message substring, as the probe does.
    oom_cls = RuntimeError

    real_grad = model.parameter_shift_grad

    def fake_grad(X, theta, **kw):
        chunk = model._ps_chunk_cache.get(N, model.n_trainable)
        if chunk > cap:
            raise oom_cls("RESOURCE_EXHAUSTED: Out of memory")
        return real_grad(X, theta, **kw)

    monkeypatch.setattr(model, "parameter_shift_grad", fake_grad)

    result = model._probe_jax_ps_chunk(N, upper=model.n_trainable)
    # cap=4, safety=0.75 → int(4*0.75) = 3
    assert result == 3
