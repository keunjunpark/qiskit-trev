"""Parity tests: QMLModel JAX backend matches torch path.

Land with plan/14 Track A extension — QMLModel now supports the JAX
backend for forward, parameter-shift gradient, and population forward.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from qiskit.circuit import QuantumCircuit

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from qiskit_trev.qml import QMLModel


def _make_circuit(n_qubits: int = 4, n_layers: int = 1):
    qc = QuantumCircuit(n_qubits)
    data_indices: list[int] = []
    trainable_indices: list[int] = []
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(0.0, q)
            data_indices.append(idx)
            idx += 1
        for q in range(n_qubits):
            qc.ry(0.0, q)
            trainable_indices.append(idx)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc, data_indices, trainable_indices


def _xy(n_samples: int, n_features: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    return torch.from_numpy(X)


def _theta(P: int, seed: int = 1):
    rng = np.random.RandomState(seed)
    return torch.from_numpy((rng.rand(P) * 6.28).astype(np.float32))


# ---------------------------------------------------------------- forward

@pytest.mark.parametrize("n_qubits,n_layers,N", [(3, 1, 4), (4, 1, 8), (4, 2, 6)])
def test_forward_matches_torch(n_qubits, n_layers, N):
    qc, di, ti = _make_circuit(n_qubits, n_layers)
    torch_model = QMLModel(qc, di, ti, rank=4, device="cpu", backend="torch")
    jax_model = QMLModel(qc, di, ti, rank=4, device="cpu", backend="jax")

    X = _xy(N, n_qubits)
    theta = _theta(len(ti))

    out_t = torch_model(X, theta).cpu().numpy()
    out_j = jax_model(X, theta).cpu().numpy()

    # Same magnitude & sign; fp32 + SVD gauge budget from build_batch path.
    np.testing.assert_allclose(out_j, out_t, atol=5e-3, rtol=5e-2)


def test_forward_env_var_forces_jax(monkeypatch):
    monkeypatch.setenv("QISKIT_TREV_BACKEND", "jax")
    qc, di, ti = _make_circuit(3, 1)
    model = QMLModel(qc, di, ti, rank=4, device="cpu")  # no backend arg
    X = _xy(4, 3)
    theta = _theta(len(ti))
    out = model(X, theta)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 4)


def test_forward_jax_array_input_returns_jax():
    """Passing a jax.Array for theta gets a jax.Array back (auto dispatch)."""
    qc, di, ti = _make_circuit(3, 1)
    model = QMLModel(qc, di, ti, rank=4, device="cpu")

    X_j = jnp.asarray(_xy(4, 3).numpy())
    theta_j = jnp.asarray(_theta(len(ti)).numpy())
    out = model(X_j, theta_j)
    assert type(out).__module__.startswith(("jax", "jaxlib"))
    assert out.shape == (3, 4)


# ---------------------------------------------------------------- grad

@pytest.mark.parametrize("n_qubits,n_layers,N", [(3, 1, 3), (4, 1, 5)])
def test_parameter_shift_grad_matches_torch(n_qubits, n_layers, N):
    qc, di, ti = _make_circuit(n_qubits, n_layers)
    torch_model = QMLModel(qc, di, ti, rank=4, device="cpu", backend="torch")
    jax_model = QMLModel(qc, di, ti, rank=4, device="cpu", backend="jax")

    X = _xy(N, n_qubits)
    theta = _theta(len(ti))

    g_t = torch_model.parameter_shift_grad(X, theta).cpu().numpy()
    g_j = jax_model.parameter_shift_grad(X, theta).cpu().numpy()

    np.testing.assert_allclose(g_j, g_t, atol=5e-3, rtol=5e-2)


# ---------------------------------------------------------------- population

def test_forward_population_matches_torch():
    qc, di, ti = _make_circuit(3, 1)
    torch_model = QMLModel(qc, di, ti, rank=4, device="cpu", backend="torch")
    jax_model = QMLModel(qc, di, ti, rank=4, device="cpu", backend="jax")

    X = _xy(4, 3)
    pop_thetas = torch.stack([_theta(len(ti), seed=s) for s in range(3)])

    p_t = torch_model.forward_population(X, pop_thetas).cpu().numpy()
    p_j = jax_model.forward_population(X, pop_thetas).cpu().numpy()

    np.testing.assert_allclose(p_j, p_t, atol=5e-3, rtol=5e-2)


# ---------------------------------------------------------------- toggle

def test_invalid_backend_raises():
    qc, di, ti = _make_circuit(2, 1)
    with pytest.raises(ValueError, match="backend must be one of"):
        QMLModel(qc, di, ti, rank=4, device="cpu", backend="nonsense")


def test_jit_cache_reuse_on_same_shape():
    """Second call with the same shapes reuses the compiled kernel."""
    qc, di, ti = _make_circuit(3, 1)
    model = QMLModel(qc, di, ti, rank=4, device="cpu", backend="jax")

    X = _xy(4, 3)
    theta1 = _theta(len(ti), seed=0)
    theta2 = _theta(len(ti), seed=1)
    model(X, theta1)
    model(X, theta2)

    # Exactly one cache entry for this (shape, forward) key.
    forward_keys = [k for k in model._jax_jit_cache if k[0] == "forward"]
    assert len(forward_keys) == 1
