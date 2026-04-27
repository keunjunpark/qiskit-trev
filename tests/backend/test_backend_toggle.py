"""Tests for the backend toggle (env var + per-object flag)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from qiskit_trev.gradient import BatchParameterShiftGradient, _resolve_backend_pref
from qiskit_trev.model import TensorRingModel


def _tiny_model(N: int = 4, reps: int = 1, rank: int = 4):
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


# --- _resolve_backend_pref -------------------------------------------------

def test_resolve_pref_arg_wins(monkeypatch):
    monkeypatch.setenv("QISKIT_TREV_BACKEND", "jax")
    assert _resolve_backend_pref("torch") == "torch"


def test_resolve_pref_env_wins_when_no_arg(monkeypatch):
    monkeypatch.setenv("QISKIT_TREV_BACKEND", "jax")
    assert _resolve_backend_pref(None) == "jax"


def test_resolve_pref_env_default_is_torch(monkeypatch):
    monkeypatch.delenv("QISKIT_TREV_BACKEND", raising=False)
    assert _resolve_backend_pref(None) == "torch"


def test_resolve_pref_env_case_insensitive(monkeypatch):
    monkeypatch.setenv("QISKIT_TREV_BACKEND", "JAX")
    assert _resolve_backend_pref(None) == "jax"


def test_resolve_pref_invalid_env_warns(monkeypatch):
    monkeypatch.setenv("QISKIT_TREV_BACKEND", "bogus")
    with pytest.warns(RuntimeWarning, match="QISKIT_TREV_BACKEND"):
        assert _resolve_backend_pref(None) == "torch"


def test_constructor_rejects_invalid_backend():
    model, _ = _tiny_model()
    with pytest.raises(ValueError, match="backend must be one of"):
        BatchParameterShiftGradient(model, backend="not-a-backend")


# --- Per-object forcing ----------------------------------------------------

def test_force_jax_converts_torch_input_and_returns_torch():
    """User passes torch params with backend='jax' — JAX compute, torch out."""
    model, P = _tiny_model(N=4, reps=1, rank=4)
    grad_fn = BatchParameterShiftGradient(model, backend="jax")

    params = torch.rand(P) * 6.28
    g = grad_fn(params)

    assert isinstance(g, torch.Tensor)
    assert g.shape == (P,)

    # Matches the pure-torch backend within the usual tolerance.
    grad_fn_torch = BatchParameterShiftGradient(model, backend="torch")
    g_ref = grad_fn_torch(params)
    torch.testing.assert_close(
        g.to(torch.float64), g_ref, atol=5e-3, rtol=5e-2
    )


def test_force_torch_on_jax_input_returns_jax():
    """User passes jax params with backend='torch' — torch compute, jax out."""
    model, P = _tiny_model(N=4, reps=1, rank=4)
    grad_fn = BatchParameterShiftGradient(model, backend="torch")

    params_j = jnp.asarray((np.random.RandomState(0).rand(P) * 6.28).astype(np.float32))
    g = grad_fn(params_j)

    # Output should be a JAX array (preserves input type).
    assert type(g).__module__.startswith(("jax", "jaxlib"))
    assert g.shape == (P,)


def test_default_is_torch(monkeypatch):
    """backend=None with no env var resolves to 'torch'. JAX is opt-in."""
    monkeypatch.delenv("QISKIT_TREV_BACKEND", raising=False)
    model, P = _tiny_model(N=4, reps=1, rank=4)
    grad_fn = BatchParameterShiftGradient(model)  # backend defaults to None

    assert _resolve_backend_pref(grad_fn._backend_arg) == "torch"

    # torch input runs the torch path naturally.
    g_t = grad_fn(torch.rand(P) * 6.28)
    assert isinstance(g_t, torch.Tensor)

    # jax input: torch path runs the compute, but return type still
    # mirrors the input type (existing convention, see test_force_torch_*).
    g_j = grad_fn(jnp.asarray(np.random.rand(P).astype(np.float32) * 6.28))
    assert type(g_j).__module__.startswith(("jax", "jaxlib"))


def test_explicit_auto_dispatches_by_input_type():
    """backend='auto' still dispatches by input type for users who opt in."""
    model, P = _tiny_model(N=4, reps=1, rank=4)
    grad_fn = BatchParameterShiftGradient(model, backend="auto")

    g_t = grad_fn(torch.rand(P) * 6.28)
    assert isinstance(g_t, torch.Tensor)

    g_j = grad_fn(jnp.asarray(np.random.rand(P).astype(np.float32) * 6.28))
    assert type(g_j).__module__.startswith(("jax", "jaxlib"))


def test_env_var_forces_jax_on_torch_input(monkeypatch):
    """Setting QISKIT_TREV_BACKEND=jax without changing code uses JAX path."""
    monkeypatch.setenv("QISKIT_TREV_BACKEND", "jax")
    model, P = _tiny_model(N=4, reps=1, rank=4)
    grad_fn = BatchParameterShiftGradient(model)  # no backend arg

    params = torch.rand(P) * 6.28
    g = grad_fn(params)
    assert isinstance(g, torch.Tensor)
    assert g.shape == (P,)
