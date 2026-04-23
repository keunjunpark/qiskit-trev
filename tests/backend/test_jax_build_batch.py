"""Parity tests: JAX build_batch + full forward matches torch.

Land with plan/14 Step 3. Covers:
  - ``TensorRingState.build_batch`` dispatches on JAX input and matches
    the torch output.
  - End-to-end forward (build + expectation value) matches torch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from qiskit_trev.hamiltonian import Hamiltonian
from qiskit_trev.measure.efficient_contraction import batched_expectation_value
from qiskit_trev.tensor_ring.state import GateInstruction, TensorRingState


def _make_circuit(N: int, reps: int = 2):
    """Simple parameterized circuit — RY layer + CNOT ring, repeated."""
    gates: list[GateInstruction] = []
    p = 0
    for _ in range(reps):
        for q in range(N):
            gates.append(GateInstruction("RY", (q,), params=(0.0,),
                                         param_indices=(p,)))
            p += 1
        for q in range(N):
            gates.append(GateInstruction("CNOT", (q, (q + 1) % N)))
    return gates, p


def _torch_params(P: int, B: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, P, generator=g) * 2 * 3.14159


def _make_ham(N: int, C: int, seed: int = 1) -> Hamiltonian:
    rng = np.random.RandomState(seed)
    terms: list[tuple[str, complex]] = []
    for _ in range(C):
        s = "".join(rng.choice(["I", "Z"]) for _ in range(N))
        coef = float(rng.rand() - 0.5)
        terms.append((s, coef))
    if C >= 1 and "Z" not in terms[0][0]:
        terms[0] = ("Z" + terms[0][0][1:], terms[0][1])
    return Hamiltonian.from_pauli_list(terms)


GRID = [
    (6, 4, 1, 2),
    (6, 4, 4, 2),
    (8, 6, 4, 2),
    (10, 8, 4, 2),
]


@pytest.mark.parametrize("N,chi,B,reps", GRID)
def test_full_forward_matches_torch_highest(N, chi, B, reps):
    """build_batch + batched_expectation_value end-to-end, HIGHEST precision.

    Element-wise parity on ``build_batch`` output itself is not meaningful —
    tensor-ring states are defined up to a bond-gauge freedom, so the torch
    and JAX SVD truncations can land on different representations of the
    same physical state. Expectation values are gauge-invariant and are
    the right thing to test.
    """
    from qiskit_trev.backend import JAX_BACKEND_HIGHEST
    from qiskit_trev.measure.efficient_contraction import (
        _batched_expectation_kernel,
    )
    from qiskit_trev.tensor_ring._state_jax import build_batch_jax

    gates_t, P = _make_circuit(N, reps=reps)
    gates_j, _ = _make_circuit(N, reps=reps)
    params_t = _torch_params(P, B)
    ham = _make_ham(N, C=3)

    trs = TensorRingState(num_qubits=N, rank=chi)
    torch_state = trs.build_batch(gates_t, params_t)
    torch_out = batched_expectation_value(torch_state, ham).cpu().numpy()

    params_j = jnp.asarray(params_t.numpy())
    jax_state = build_batch_jax(
        N, chi, gates_j, params_j, precision=jax.lax.Precision.HIGHEST
    )
    paulis = jnp.asarray(ham.get_bool_pauli_tensor().numpy())
    coeffs = jnp.asarray(np.asarray(ham.coefficients, dtype=np.complex64))
    Z_op = jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex64)
    I_op = jnp.eye(2, dtype=jnp.complex64)
    total_init = jnp.zeros(B, dtype=jnp.complex64)

    jax_out = np.asarray(
        _batched_expectation_kernel(
            JAX_BACKEND_HIGHEST,
            jax_state,
            paulis,
            coeffs,
            Z_op,
            I_op,
            total_init,
            None,
        )
    )

    # SVD truncation at small bond dim can pick slightly different valid
    # low-rank representations between torch and JAX. Expectation values
    # remain close but not bit-identical — tolerance reflects reality, not
    # the underlying einsum/matmul precision (which is tight at HIGHEST).
    np.testing.assert_allclose(jax_out, torch_out, atol=2e-3, rtol=5e-2)


@pytest.mark.parametrize("N,chi,B,reps", GRID)
def test_full_forward_matches_torch_default(N, chi, B, reps):
    """End-to-end at DEFAULT precision (what benchmarks see)."""
    gates_t, P = _make_circuit(N, reps=reps)
    gates_j, _ = _make_circuit(N, reps=reps)
    params_t = _torch_params(P, B)
    ham = _make_ham(N, C=3)

    trs = TensorRingState(num_qubits=N, rank=chi)
    torch_state = trs.build_batch(gates_t, params_t)
    torch_out = batched_expectation_value(torch_state, ham).cpu().numpy()

    params_j = jnp.asarray(params_t.numpy())
    jax_state = trs.build_batch(gates_j, params_j)  # dispatches to JAX DEFAULT
    jax_out = np.asarray(batched_expectation_value(jax_state, ham))

    # DEFAULT precision: tf32-like matmul + SVD drift compound.
    np.testing.assert_allclose(jax_out, torch_out, atol=5e-3, rtol=1e-1)


def test_dispatch_returns_jax_array_on_jax_input():
    """build_batch returns jax.Array when given jax params."""
    gates, P = _make_circuit(4, reps=1)
    params_j = jnp.asarray(_torch_params(P, 2).numpy())
    trs = TensorRingState(num_qubits=4, rank=4)
    out = trs.build_batch(gates, params_j)
    assert "jax" in type(out).__module__
    assert out.shape == (2, 4, 4, 4, 2)


def test_dispatch_returns_torch_tensor_on_torch_input():
    """build_batch stays torch-native on torch params."""
    gates, P = _make_circuit(4, reps=1)
    params_t = _torch_params(P, 2)
    trs = TensorRingState(num_qubits=4, rank=4)
    out = trs.build_batch(gates, params_t)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 4, 4, 4, 2)


def _make_zz_ring_circuit(N: int, reps: int = 1):
    """RY layer + ZZ ring, repeated — the QAOA/HVA-style ansatz.

    Exercises the parameterised-2q-run fori_loop path (``ZZ``). The last
    ZZ on each layer wraps (q=N-1, q+1=0) and falls through to
    single-gate handling; the first N-1 ZZs on each layer form the
    coalesced run.
    """
    gates: list[GateInstruction] = []
    p = 0
    for _ in range(reps):
        for q in range(N):
            gates.append(
                GateInstruction("RY", (q,), params=(0.0,), param_indices=(p,))
            )
            p += 1
        for q in range(N):
            gates.append(
                GateInstruction(
                    "ZZ", (q, (q + 1) % N), params=(0.0,), param_indices=(p,)
                )
            )
            p += 1
    return gates, p


@pytest.mark.parametrize("N,chi,B,reps", [(4, 4, 2, 1), (6, 6, 2, 1), (6, 6, 2, 2)])
def test_zz_ring_forward_matches_torch(N, chi, B, reps):
    """End-to-end forward on a ZZ-ring ansatz: JAX (via 2q_run_param
    fori_loop) must match the torch reference path."""
    from qiskit_trev.backend import JAX_BACKEND_HIGHEST
    from qiskit_trev.measure.efficient_contraction import (
        _batched_expectation_kernel,
    )
    from qiskit_trev.tensor_ring._state_jax import build_batch_jax

    gates_t, P = _make_zz_ring_circuit(N, reps=reps)
    gates_j, _ = _make_zz_ring_circuit(N, reps=reps)
    params_t = _torch_params(P, B)
    ham = _make_ham(N, C=3)

    trs = TensorRingState(num_qubits=N, rank=chi)
    torch_state = trs.build_batch(gates_t, params_t)
    torch_out = batched_expectation_value(torch_state, ham).cpu().numpy()

    params_j = jnp.asarray(params_t.numpy())
    jax_state = build_batch_jax(
        N, chi, gates_j, params_j, precision=jax.lax.Precision.HIGHEST
    )
    paulis = jnp.asarray(ham.get_bool_pauli_tensor().numpy())
    coeffs = jnp.asarray(np.asarray(ham.coefficients, dtype=np.complex64))
    Z_op = jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex64)
    I_op = jnp.eye(2, dtype=jnp.complex64)
    total_init = jnp.zeros(B, dtype=jnp.complex64)

    jax_out = np.asarray(
        _batched_expectation_kernel(
            JAX_BACKEND_HIGHEST, jax_state, paulis, coeffs,
            Z_op, I_op, total_init, None,
        )
    )

    # Same tolerance as the RY+CNOT test — gauge freedom + fp32 + SVD.
    np.testing.assert_allclose(jax_out, torch_out, atol=2e-3, rtol=5e-2)


def test_zz_run_is_coalesced():
    """Sanity: the coalescer actually produces a 2q_run_param entry for
    the ZZ-ring ansatz, so the new fori_loop path gets exercised."""
    from qiskit_trev.tensor_ring._state_jax import (
        _coalesce_2q_runs,
        _compile_fused_ops,
    )

    gates, _ = _make_zz_ring_circuit(N=6, reps=1)
    ops = _coalesce_2q_runs(_compile_fused_ops(gates), N=6)
    run_tags = [op[0] for op in ops]
    assert "2q_run_param" in run_tags, (
        f"expected a parameterised 2q run, got ops: {run_tags}"
    )
