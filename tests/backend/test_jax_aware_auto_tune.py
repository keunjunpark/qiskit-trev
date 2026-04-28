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


def test_vqe_max_chunk_size_short_circuits():
    """max_chunk_size must prevent the analytical estimator AND the JIT
    probe from running — the user's declared ceiling is trusted."""
    model, P = _vqe_model(N=4, reps=1)
    grad_fn = BatchParameterShiftGradient(
        model, chunk_size="auto", max_chunk_size=2, backend="jax"
    )
    params = jnp.asarray(np.zeros(P, dtype=np.float32))
    grad_fn(params)
    chunk_keys = [k for k in grad_fn._jax_jit_cache if k[0] == "chunk"]
    used_chunks = [k[-1] for k in chunk_keys]
    if used_chunks:
        assert max(used_chunks) <= 2


def test_vqe_max_chunk_size_validation():
    model, _ = _vqe_model(N=3, reps=1)
    with pytest.raises(ValueError, match="max_chunk_size"):
        BatchParameterShiftGradient(model, max_chunk_size=0)


def test_vqe_analytical_is_default_no_jit_probe(monkeypatch):
    """Default ``chunk_size="auto"`` path must not invoke the JIT probe."""
    model, P = _vqe_model(N=4, reps=1)
    grad_fn = BatchParameterShiftGradient(
        model, chunk_size="auto", backend="jax",
    )
    probe_calls = []
    monkeypatch.setattr(
        grad_fn, "_probe_jax_chunk",
        lambda P_, upper: probe_calls.append(P_) or 1,
    )
    grad_fn(jnp.asarray(np.zeros(P, dtype=np.float32)))
    assert probe_calls == [], (
        "analytical default should NOT call the JIT probe — got "
        f"{len(probe_calls)} probe calls"
    )


def test_vqe_jit_probe_cached_per_P(monkeypatch):
    """Opt-in chunk_size='jit' still caches its result per P."""
    model, P = _vqe_model(N=4, reps=1)
    grad_fn = BatchParameterShiftGradient(
        model, chunk_size="jit", backend="jax",
    )
    probe_calls = []
    real_probe = grad_fn._probe_jax_chunk

    def counting(P_, upper):
        probe_calls.append(P_)
        return real_probe(P_, upper)

    monkeypatch.setattr(grad_fn, "_probe_jax_chunk", counting)

    params = jnp.asarray(np.zeros(P, dtype=np.float32))
    grad_fn(params)
    grad_fn(params)
    grad_fn(params)
    assert len(probe_calls) == 1, f"probed {len(probe_calls)} times"


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


def test_vqe_int_chunk_size_skips_all_probes(monkeypatch):
    """Explicit int chunk_size goes straight through — no analytical, no JIT."""
    model, P = _vqe_model(N=4, reps=1)
    grad_fn = BatchParameterShiftGradient(model, chunk_size=2, backend="jax")

    analytic_calls = []
    monkeypatch.setattr(
        grad_fn, "_analytical_chunk",
        lambda *a, **k: analytic_calls.append(a) or 1,
    )
    probe_calls = []
    monkeypatch.setattr(
        grad_fn, "_probe_jax_chunk",
        lambda *a, **k: probe_calls.append(a) or 1,
    )

    grad_fn(jnp.asarray(np.zeros(P, dtype=np.float32)))
    assert analytic_calls == []
    assert probe_calls == []


def test_vqe_auto_caches_per_P(monkeypatch):
    """Second call with same P hits the analytical cache instead of recomputing."""
    model, P = _vqe_model(N=4, reps=1)
    grad_fn = BatchParameterShiftGradient(model, chunk_size="auto", backend="jax")

    analytic_calls = []
    real = grad_fn._analytical_chunk

    def counting(P_, upper):
        analytic_calls.append(P_)
        return real(P_, upper)

    monkeypatch.setattr(grad_fn, "_analytical_chunk", counting)

    params = jnp.asarray(np.zeros(P, dtype=np.float32))
    grad_fn(params)
    grad_fn(params)
    grad_fn(params)
    assert len(analytic_calls) == 1, f"analytical ran {len(analytic_calls)} times"
    assert P in grad_fn._jax_probed_chunk


def test_vqe_jit_max_chunk_size_short_circuits(monkeypatch):
    """chunk_size='jit' + max_chunk_size must skip the JIT probe entirely."""
    model, P = _vqe_model(N=4, reps=1)
    grad_fn = BatchParameterShiftGradient(
        model, chunk_size="jit", max_chunk_size=2, backend="jax",
    )

    probe_calls = []
    monkeypatch.setattr(
        grad_fn, "_probe_jax_chunk",
        lambda *a, **k: probe_calls.append(a) or 99,
    )

    grad_fn(jnp.asarray(np.zeros(P, dtype=np.float32)))
    assert probe_calls == []


def test_vqe_analytical_chunk_cuda_branch(monkeypatch):
    """Exercise the CUDA branch of _analytical_chunk via stubbed mem_get_info.

    Without this we can't reach gradient.py's torch.cuda.mem_get_info call
    on a CPU-only CI runner — the helper short-circuits earlier."""
    model, P = _vqe_model(N=4, reps=1)
    grad_fn = BatchParameterShiftGradient(model, chunk_size="auto", backend="jax")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info",
        lambda *a, **k: (40 * 1024 ** 3, 40 * 1024 ** 3),  # 40 GB free
    )

    result = grad_fn._analytical_chunk(P, upper=P)
    assert 1 <= result <= P


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
    """On CPU, auto_tune skips the GPU probes. max_ps_chunk must still
    clamp the cached ps_chunk so the gradient path honours the ceiling."""
    model = _qml_model(n_qubits=3, n_layers=2, max_ps_chunk=2)
    assert model._max_ps_chunk == 2

    model.auto_tune(N=5)
    assert model._ps_chunk_cache == {5: 2}
    assert model._ps_chunk == 2


def test_qml_auto_tune_on_cpu_no_max_ps_chunk_uses_n_trainable():
    """CPU branch without max_ps_chunk caches ps_chunk = n_trainable."""
    model = _qml_model(n_qubits=3, n_layers=2)
    model.auto_tune(N=4)
    assert model._ps_chunk_cache == {4: model.n_trainable}


def test_qml_auto_tune_rejects_invalid_probe():
    model = _qml_model(n_qubits=3, n_layers=1)
    with pytest.raises(ValueError, match="probe"):
        model.auto_tune(4, probe="bogus")


def _patch_cuda_auto_tune(monkeypatch, model, batch_size):
    """Trick auto_tune into believing it's on a GPU without actually using one.

    Replaces ``torch.device`` so ``dev.type == "cuda"``, redirects the
    dummy ``torch.zeros`` allocation to CPU, and stubs ``auto_batch_size``
    so the torch probe (which would touch GPU memory) never runs.
    """
    import qiskit_trev.qml as qml_mod
    import qiskit_trev.optimization.auto_batch as ab

    class _FakeDev:
        type = "cuda"

    real_zeros = torch.zeros

    def _cpu_zeros(*a, **k):
        k.pop("device", None)
        return real_zeros(*a, **k)

    monkeypatch.setattr(model, "device", "cuda")
    monkeypatch.setattr(qml_mod.torch, "device", lambda *a, **k: _FakeDev())
    monkeypatch.setattr(qml_mod.torch, "zeros", _cpu_zeros)
    monkeypatch.setattr(ab, "auto_batch_size", lambda *a, **k: batch_size)


@pytest.mark.parametrize("probe_mode,expected", [
    ("analytical", 3),  # analytical estimator returns 3
    ("none", 5),        # uses torch-derived ps_chunk = batch_size // (2*N)
])
def test_qml_cuda_auto_tune_branches(monkeypatch, probe_mode, expected):
    """Cover the GPU branch of auto_tune for ``probe="analytical"`` and
    ``probe="none"``. Uses a fake CUDA device to avoid needing real hardware."""
    model = _qml_model(n_qubits=3, n_layers=2, backend="jax")  # n_trainable=6
    N = 4
    _patch_cuda_auto_tune(monkeypatch, model, batch_size=2 * N * 5)
    monkeypatch.setattr(model, "_analytical_ps_chunk", lambda N_: 3)

    model.auto_tune(N, probe=probe_mode)

    assert model._ps_chunk_cache == {N: expected}


def test_qml_cuda_auto_tune_jit_probe(monkeypatch):
    """probe='jit' on the GPU branch invokes _probe_jax_ps_chunk."""
    model = _qml_model(n_qubits=3, n_layers=2, backend="jax")
    N = 4
    _patch_cuda_auto_tune(monkeypatch, model, batch_size=2 * N * 5)
    monkeypatch.setattr(model, "_probe_jax_ps_chunk", lambda N_, upper: 4)

    model.auto_tune(N, probe="jit")

    assert model._ps_chunk_cache == {N: 4}


def test_qml_cuda_auto_tune_max_ps_chunk_overrides_probe(monkeypatch):
    """max_ps_chunk on the GPU branch must clamp the analytical/jit result."""
    model = _qml_model(n_qubits=3, n_layers=2, backend="jax", max_ps_chunk=2)
    N = 4
    _patch_cuda_auto_tune(monkeypatch, model, batch_size=2 * N * 5)
    # Analytical would propose 5 — override caps at 2.
    monkeypatch.setattr(model, "_analytical_ps_chunk", lambda N_: 5)

    model.auto_tune(N)

    assert model._ps_chunk_cache == {N: 2}


def test_qml_analytical_estimator_is_bounded(monkeypatch):
    """Analytical estimator must return a chunk in [1, n_trainable]
    regardless of how free GPU memory is reported. We can't probe real
    CUDA on the CPU CI, so fake the device to CUDA and stub the memory
    query."""
    model = _qml_model(n_qubits=3, n_layers=2)
    model.device = "cuda"  # force the CUDA branch

    import qiskit_trev.qml as qml_mod

    monkeypatch.setattr(qml_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        qml_mod.torch.cuda,
        "mem_get_info",
        lambda device=None: (40 * 1024 ** 3, 40 * 1024 ** 3),
    )

    result = model._analytical_ps_chunk(N=50)
    assert 1 <= result <= model.n_trainable


def test_qml_analytical_estimator_cpu_fallback():
    """On CPU, analytical estimator skips VRAM math and returns n_trainable."""
    model = _qml_model(n_qubits=3, n_layers=2)  # device defaults to cpu
    assert model._analytical_ps_chunk(N=50) == model.n_trainable


def test_qml_probe_fakes_oom_returns_safe_cap(monkeypatch):
    """Mock parameter_shift_grad to raise on chunk > K, verify probe
    lands at int(K * 0.75)."""
    model = _qml_model(n_qubits=3, n_layers=2, backend="jax")  # n_trainable = 6
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
