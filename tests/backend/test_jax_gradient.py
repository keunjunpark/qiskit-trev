"""Parity tests: JAX parameter-shift gradient matches torch.

Land with plan/14 Step 4.
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


def _build_model(N: int, reps: int, rank: int) -> tuple[TensorRingModel, int]:
    """RY + CNOT-ring ansatz. ``qc.ry(0.0, q)`` uses float placeholders;
    the real param values are fed through the gradient call site, matching
    how the rest of the test suite constructs models.
    """
    qc = QuantumCircuit(N)
    P = 0
    for _ in range(reps):
        for q in range(N):
            qc.ry(0.0, q)
            P += 1
        for q in range(N):
            qc.cx(q, (q + 1) % N)

    obs = SparsePauliOp.from_list(
        [("Z" + "I" * (N - 1), 1.0), ("I" + "Z" + "I" * (N - 2), 0.5)]
    )
    model = TensorRingModel(qc, obs, rank=rank)
    return model, P


GRID = [
    (4, 1, 4),
    (6, 1, 6),
    (8, 1, 8),
]


@pytest.mark.parametrize("N,reps,rank", GRID)
def test_gradient_matches_torch(N, reps, rank):
    model, P = _build_model(N, reps, rank)
    rng = np.random.RandomState(0)
    params_np = (rng.rand(P) * 2 * math.pi).astype(np.float32)

    grad_fn = BatchParameterShiftGradient(model)
    grad_torch = grad_fn(torch.from_numpy(params_np.astype(np.float64))).cpu().numpy()

    grad_jax = np.asarray(grad_fn(jnp.asarray(params_np)))

    # Gradient entries can be near zero (flat regions) — use a moderate
    # absolute floor alongside relative tolerance. SVD truncation + fp32
    # accumulate here more than in forward alone.
    np.testing.assert_allclose(grad_jax, grad_torch, atol=5e-3, rtol=5e-2)


def test_gradient_dispatch_returns_jax_on_jax_input():
    model, P = _build_model(4, 1, 4)
    params_j = jnp.asarray(np.zeros(P, dtype=np.float32))
    grad_fn = BatchParameterShiftGradient(model)
    out = grad_fn(params_j)
    assert "jax" in type(out).__module__
    assert out.shape == (P,)


def test_gradient_dispatch_returns_torch_on_torch_input():
    model, P = _build_model(4, 1, 4)
    params_t = torch.zeros(P, dtype=torch.float64)
    grad_fn = BatchParameterShiftGradient(model)
    out = grad_fn(params_t)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (P,)


def test_gradient_jit_cache_hits_on_same_shape():
    """Second call with same-shape params reuses the compiled kernel."""
    model, P = _build_model(4, 1, 4)
    grad_fn = BatchParameterShiftGradient(model)

    p1 = jnp.asarray(np.random.RandomState(0).rand(P).astype(np.float32))
    p2 = jnp.asarray(np.random.RandomState(1).rand(P).astype(np.float32))
    grad_fn(p1)
    grad_fn(p2)

    # Cache should have exactly one entry for this (model, shift, P).
    assert len(grad_fn._jax_jit_cache) == 1
