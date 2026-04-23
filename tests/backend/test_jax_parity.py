"""Parity tests: JAX backend matches torch on batched_expectation_value.

Land with plan/14 Step 2. Both backends are run on the same workload; torch
is the source of truth, JAX must match within 1e-5 abs / 1e-4 rel.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from qiskit_trev.hamiltonian import Hamiltonian
from qiskit_trev.measure.efficient_contraction import (
    _batched_expectation_kernel,
    batched_expectation_value,
)


def _make_ham(N: int, C: int, seed: int = 0) -> Hamiltonian:
    rng = np.random.RandomState(seed)
    terms: list[tuple[str, complex]] = []
    for i in range(C):
        s = "".join(rng.choice(["I", "Z"]) for _ in range(N))
        coef = float(rng.rand() - 0.5)
        terms.append((s, coef))
    # Guarantee at least one Z so tests exercise the Z_op path.
    if C >= 1 and "Z" not in terms[0][0]:
        terms[0] = ("Z" + terms[0][0][1:], terms[0][1])
    return Hamiltonian.from_pauli_list(terms)


def _torch_batch(B: int, N: int, chi: int, seed: int = 42) -> torch.Tensor:
    """Random tensor-ring batch, scaled so contracted norms stay O(1).

    Without scaling, accumulation grows like chi^N and float32 rounding
    produces ~1e-3 relative error between backends — fine for the physics,
    wrong for a parity test.
    """
    gen = torch.Generator().manual_seed(seed)
    t = torch.randn(B, N, chi, chi, 2, dtype=torch.cfloat, generator=gen)
    return t / (chi ** 0.5)


GRID = [
    (4, 6, 1, 1),
    (4, 6, 4, 1),
    (4, 6, 10, 4),
    (8, 12, 10, 4),
    (8, 12, 1, 8),
    (12, 16, 5, 2),
]


def _jax_constants(ham: Hamiltonian, B: int):
    paulis = jnp.asarray(ham.get_bool_pauli_tensor().numpy())
    coeffs = jnp.asarray(np.asarray(ham.coefficients, dtype=np.complex64))
    Z_op = jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex64)
    I_op = jnp.eye(2, dtype=jnp.complex64)
    total_init = jnp.zeros(B, dtype=jnp.complex64)
    return paulis, coeffs, Z_op, I_op, total_init


@pytest.mark.parametrize("chi,N,C,B", GRID)
def test_batched_expectation_value_matches_torch(chi, N, C, B):
    """JAX backend at HIGHEST precision is bit-identical (within fp32) to torch.

    At ``Precision.DEFAULT`` JAX uses tf32-like reduced-mantissa matmul on
    GPU and drifts ~1e-4 relative; ``HIGHEST`` closes the gap. See the
    backend docstring for the trade-off.
    """
    from qiskit_trev.backend import JAX_BACKEND_HIGHEST

    ham = _make_ham(N, C)
    torch_batch = _torch_batch(B, N, chi)
    expected = batched_expectation_value(torch_batch, ham).cpu().numpy()

    jax_batch = jnp.asarray(torch_batch.numpy())
    paulis, coeffs, Z_op, I_op, total = _jax_constants(ham, B)
    actual = np.asarray(
        _batched_expectation_kernel(
            JAX_BACKEND_HIGHEST, jax_batch, paulis, coeffs, Z_op, I_op, total, None
        )
    )

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("chi,N,C,B", GRID)
def test_batched_expectation_value_default_precision_within_tolerance(chi, N, C, B):
    """DEFAULT precision (the one benchmarks use) stays within fp32-tf32 budget."""
    ham = _make_ham(N, C)
    torch_batch = _torch_batch(B, N, chi)
    expected = batched_expectation_value(torch_batch, ham).cpu().numpy()

    jax_batch = jnp.asarray(torch_batch.numpy())
    actual = np.asarray(batched_expectation_value(jax_batch, ham))

    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=5e-2)


@pytest.mark.parametrize("chi,N,C,B", GRID)
def test_jitted_kernel_matches_torch(chi, N, C, B):
    from qiskit_trev.backend import JAX_BACKEND_HIGHEST

    ham = _make_ham(N, C)
    torch_batch = _torch_batch(B, N, chi)
    expected = batched_expectation_value(torch_batch, ham).cpu().numpy()

    jax_batch = jnp.asarray(torch_batch.numpy())
    paulis, coeffs, Z_op, I_op, total_init = _jax_constants(ham, B)

    jit_kernel = jax.jit(_batched_expectation_kernel, static_argnums=(0, 7))
    actual = np.asarray(
        jit_kernel(
            JAX_BACKEND_HIGHEST, jax_batch, paulis, coeffs, Z_op, I_op, total_init, None
        )
    )

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_jit_caches_across_same_shape_calls():
    """Same-shape calls reuse the compiled kernel (no recompile)."""
    from qiskit_trev.backend import JAX_BACKEND

    ham = _make_ham(6, 3)
    paulis, coeffs, Z_op, I_op, _ = _jax_constants(ham, B=4)

    jit_kernel = jax.jit(_batched_expectation_kernel, static_argnums=(0, 7))
    initial_cache = jit_kernel._cache_size()

    # Three calls at one shape, then a fourth at a different shape.
    batch_a = jnp.asarray(_torch_batch(4, 6, 4).numpy())
    total_a = jnp.zeros(4, dtype=jnp.complex64)
    for _ in range(3):
        jit_kernel(
            JAX_BACKEND, batch_a, paulis, coeffs, Z_op, I_op, total_a, None
        ).block_until_ready()

    batch_b = jnp.asarray(_torch_batch(8, 6, 4).numpy())
    total_b = jnp.zeros(8, dtype=jnp.complex64)
    jit_kernel(
        JAX_BACKEND, batch_b, paulis, coeffs, Z_op, I_op, total_b, None
    ).block_until_ready()

    # One compile per distinct shape (two shapes above).
    added = jit_kernel._cache_size() - initial_cache
    assert added == 2, f"expected 2 new compiles, got {added}"
