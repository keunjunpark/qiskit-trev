"""Cover the single-gate branches in _apply_two_qubit_gate.

RY+CNOT rings go through the coalesced fori_loop fast path; the four
branches in the single-gate fallback (`is_wrap_fwd`, `is_wrap_bwd`,
`q0<q1` non-wrap, `q0>q1` non-wrap needing swap) only fire when the
coalescer declines to group them. These tests build tiny circuits that
force each branch.
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


def _ev_match(N, gates_spec, ham_paulis, rank=4):
    """Build the state both ways and compare expectation values."""
    # Fresh gate lists per call — build_batch mutates param_indices.
    gates_t = [GateInstruction(*g) for g in gates_spec]
    gates_j = [GateInstruction(*g) for g in gates_spec]

    trs = TensorRingState(num_qubits=N, rank=rank)
    params_t = torch.zeros(1, 0)
    state_t = trs.build_batch(gates_t, params_t)

    params_j = jnp.zeros((1, 0))
    state_j = trs.build_batch(gates_j, params_j)

    ham = Hamiltonian.from_pauli_list(ham_paulis)
    ev_t = batched_expectation_value(state_t, ham).cpu().numpy()
    ev_j = np.asarray(batched_expectation_value(state_j, ham))
    np.testing.assert_allclose(ev_j, ev_t, atol=1e-3, rtol=1e-2)


def test_single_cnot_q0_lt_q1_non_wrap():
    """One CNOT between adjacent qubits with q0 < q1, not on ring wrap.
    Hits the `q0 < q1` branch in _apply_two_qubit_gate."""
    _ev_match(
        N=4,
        gates_spec=[
            ("H", (1,)),
            ("CNOT", (1, 2)),  # single gate — coalescer won't fold (len < 2)
        ],
        ham_paulis=[("ZZII", 1.0), ("IZZI", 0.5)],
    )


def test_single_cnot_q0_gt_q1_non_wrap():
    """One CNOT with q0 > q1 — forces the else-branch that swaps the gate."""
    _ev_match(
        N=4,
        gates_spec=[
            ("H", (2,)),
            ("CNOT", (2, 1)),  # q0 > q1, non-wrap → swap_gate path
        ],
        ham_paulis=[("IZII", 1.0), ("IIZI", 0.5)],
    )


def test_single_cnot_wrap_fwd():
    """CNOT(N-1, 0) — is_wrap_fwd branch."""
    _ev_match(
        N=4,
        gates_spec=[
            ("H", (3,)),
            ("CNOT", (3, 0)),
        ],
        ham_paulis=[("ZIIZ", 1.0)],
    )


def test_single_cnot_wrap_bwd():
    """CNOT(0, N-1) — is_wrap_bwd branch (swap the gate on the wrap pair)."""
    _ev_match(
        N=4,
        gates_spec=[
            ("H", (0,)),
            ("CNOT", (0, 3)),
        ],
        ham_paulis=[("ZIIZ", 1.0)],
    )


def test_single_zz_non_wrap():
    """Single parameterised 2q gate (ZZ) with theta — coalescer declines
    (len < 2), falls through to the single-gate path for parameterised
    gates. Covers the `_batched_gate_2q` param branch."""
    import math

    # Use pre-assigned param_indices so theta routes correctly.
    gates = [
        GateInstruction("H", (0,)),
        GateInstruction("H", (1,)),
        GateInstruction("ZZ", (1, 2), params=(math.pi / 3,),
                        param_indices=(0,)),
    ]
    gates_j = [
        GateInstruction("H", (0,)),
        GateInstruction("H", (1,)),
        GateInstruction("ZZ", (1, 2), params=(math.pi / 3,),
                        param_indices=(0,)),
    ]

    N = 4
    trs = TensorRingState(num_qubits=N, rank=4)
    params_t = torch.tensor([[math.pi / 3]])
    params_j = jnp.asarray(np.array([[math.pi / 3]], dtype=np.float32))

    state_t = trs.build_batch(gates, params_t)
    state_j = trs.build_batch(gates_j, params_j)

    ham = Hamiltonian.from_pauli_list([("IZZI", 1.0)])
    ev_t = batched_expectation_value(state_t, ham).cpu().numpy()
    ev_j = np.asarray(batched_expectation_value(state_j, ham))
    np.testing.assert_allclose(ev_j, ev_t, atol=1e-3, rtol=1e-2)
