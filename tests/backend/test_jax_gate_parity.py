"""Direct matrix-parity tests for every JAX gate template.

Most of _gates_jax.py's gate functions aren't exercised by the
`build_batch + expectation` end-to-end tests (which use RY + CNOT + RZZ).
These direct tests compare the JAX matrix against the torch reference
for every gate the converter supports, so ports of X/Y/Z/H/RX/RZ/U3/SWAP/
ZZ_SWAP are actually tested.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from qiskit_trev.tensor_ring import _gates_jax as gj
from qiskit_trev.tensor_ring import gates as gt


def _close(j, t, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(np.asarray(j), t.numpy(), atol=atol, rtol=rtol)


# ---------- fixed 1q gates ------------------------------------------------

@pytest.mark.parametrize(
    "name,jax_fn,torch_fn",
    [
        ("I", gj.I, gt.I),
        ("X", gj.X, gt.X),
        ("Y", gj.Y, gt.Y),
        ("Z", gj.Z, gt.Z),
        ("H", gj.H, gt.H),
    ],
)
def test_fixed_1q_matrices_match(name, jax_fn, torch_fn):
    _close(jax_fn(), torch_fn())


# ---------- parameterised 1q ---------------------------------------------

@pytest.mark.parametrize("theta", [0.0, 0.3, math.pi / 2, -1.7, math.pi])
@pytest.mark.parametrize("name,jax_fn,torch_fn", [
    ("RX", gj.RX, gt.RX),
    ("RY", gj.RY, gt.RY),
    ("RZ", gj.RZ, gt.RZ),
])
def test_rotation_scalar(name, jax_fn, torch_fn, theta):
    _close(jax_fn(theta), torch_fn(theta))


@pytest.mark.parametrize("name,jax_fn,torch_fn", [
    ("RX", gj.RX, gt.RX),
    ("RY", gj.RY, gt.RY),
    ("RZ", gj.RZ, gt.RZ),
])
def test_rotation_batched(name, jax_fn, torch_fn):
    thetas_np = np.array([0.1, 0.5, -0.3, 1.2], dtype=np.float32)
    j = jax_fn(jnp.asarray(thetas_np))
    t = torch_fn(torch.from_numpy(thetas_np))
    _close(j, t)


def test_U3_scalar():
    params = np.array([0.4, -0.7, 1.1], dtype=np.float32)
    j = gj.U3(jnp.asarray(params))
    t = gt.U3(torch.from_numpy(params))
    _close(j, t)


def test_U3_batched():
    rng = np.random.RandomState(0)
    params = (rng.rand(6, 3) * 2 * math.pi - math.pi).astype(np.float32)
    j = gj.U3(jnp.asarray(params))
    t = gt.U3(torch.from_numpy(params))
    _close(j, t)


# ---------- fixed 2q -----------------------------------------------------

@pytest.mark.parametrize("name,jax_fn,torch_fn", [
    ("CNOT", gj.CNOT, gt.CNOT),
    ("SWAP", gj.SWAP, gt.SWAP),
])
def test_fixed_2q_matrices_match(name, jax_fn, torch_fn):
    _close(jax_fn(), torch_fn())


# ---------- parameterised 2q ---------------------------------------------

@pytest.mark.parametrize("theta", [0.0, 0.3, math.pi / 2, -1.7])
@pytest.mark.parametrize("name,jax_fn,torch_fn", [
    ("ZZ", gj.ZZ, gt.ZZ),
    ("ZZ_SWAP", gj.ZZ_SWAP, gt.ZZ_SWAP),
])
def test_zz_scalar(name, jax_fn, torch_fn, theta):
    _close(jax_fn(theta), torch_fn(theta))


@pytest.mark.parametrize("name,jax_fn,torch_fn", [
    ("ZZ", gj.ZZ, gt.ZZ),
    ("ZZ_SWAP", gj.ZZ_SWAP, gt.ZZ_SWAP),
])
def test_zz_batched(name, jax_fn, torch_fn):
    thetas_np = np.array([0.1, -0.5, 1.2], dtype=np.float32)
    j = jax_fn(jnp.asarray(thetas_np))
    t = torch_fn(torch.from_numpy(thetas_np))
    _close(j, t)


# ---------- swap_gate_matrix_jax -----------------------------------------

def test_swap_gate_matrix_jax_scalar():
    """swap_gate_matrix_jax swaps qubit labels in a 4x4 gate."""
    from qiskit_trev.tensor_ring._contraction_jax import swap_gate_matrix_jax
    from qiskit_trev.tensor_ring.contraction import swap_gate_matrix

    # Use a non-symmetric 4x4 matrix so swapping actually changes something.
    rng = np.random.RandomState(0)
    M = rng.randn(4, 4).astype(np.complex64) + 1j * rng.randn(4, 4).astype(np.complex64)
    j = swap_gate_matrix_jax(jnp.asarray(M))
    t = swap_gate_matrix(torch.from_numpy(M))
    # tf32 matmul on GPU adds ~1e-3 relative noise vs torch's full fp32.
    _close(j, t, atol=1e-3, rtol=1e-3)


def test_swap_gate_matrix_jax_batched():
    from qiskit_trev.tensor_ring._contraction_jax import swap_gate_matrix_jax

    rng = np.random.RandomState(1)
    Mb = rng.randn(3, 4, 4).astype(np.complex64) + 1j * rng.randn(3, 4, 4).astype(np.complex64)
    j = swap_gate_matrix_jax(jnp.asarray(Mb))
    # Batched reference: apply scalar version to each
    from qiskit_trev.tensor_ring.contraction import swap_gate_matrix
    t = torch.stack(
        [swap_gate_matrix(torch.from_numpy(Mb[i])) for i in range(3)], dim=0
    )
    _close(j, t, atol=1e-3, rtol=1e-3)
