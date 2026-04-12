"""Tests for full contraction measurement."""

import math
import pytest
import torch
import numpy as np

from qiskit_trev.tensor_ring.state import TensorRingState, GateInstruction
from qiskit_trev.hamiltonian import Hamiltonian
from qiskit_trev.measure.full_contraction import (
    contract_tensor_ring,
    measure,
    expectation_value,
)


def _build_state(num_qubits, rank, gates):
    state = TensorRingState(num_qubits=num_qubits, rank=rank)
    return state.build(gates)


# ============================================================
# contract_tensor_ring
# ============================================================

class TestContractTensorRing:

    def test_zero_state_1q(self):
        tensor = _build_state(1, 1, [])
        amps = contract_tensor_ring(tensor)
        assert amps.shape == (2,)
        assert torch.allclose(amps, torch.tensor([1, 0], dtype=torch.cfloat), atol=1e-6)

    def test_zero_state_2q(self):
        tensor = _build_state(2, 4, [])
        amps = contract_tensor_ring(tensor)
        assert amps.shape == (2, 2)
        flat = amps.reshape(-1)
        expected = torch.tensor([1, 0, 0, 0], dtype=torch.cfloat)
        assert torch.allclose(flat, expected, atol=1e-6)

    def test_bell_state(self):
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        amps = contract_tensor_ring(tensor).reshape(-1)
        expected = torch.tensor([1, 0, 0, 1], dtype=torch.cfloat) / math.sqrt(2)
        assert torch.allclose(amps, expected, atol=1e-5)


# ============================================================
# measure (probabilities)
# ============================================================

class TestMeasure:

    def test_zero_state_probabilities(self):
        tensor = _build_state(2, 4, [])
        probs = measure(tensor)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (4,)
        np.testing.assert_allclose(probs[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(probs[1:].sum(), 0.0, atol=1e-6)

    def test_bell_state_probabilities(self):
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        probs = measure(tensor)
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-5)  # |00>
        np.testing.assert_allclose(probs[1], 0.0, atol=1e-5)  # |01>
        np.testing.assert_allclose(probs[2], 0.0, atol=1e-5)  # |10>
        np.testing.assert_allclose(probs[3], 0.5, atol=1e-5)  # |11>

    def test_probabilities_sum_to_one(self):
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("RY", (1,), params=(0.7,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        probs = measure(tensor)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)

    def test_ghz_3_qubit(self):
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
            GateInstruction("CNOT", (1, 2)),
        ]
        tensor = _build_state(3, 4, gates)
        probs = measure(tensor)
        np.testing.assert_allclose(probs[0b000], 0.5, atol=1e-4)
        np.testing.assert_allclose(probs[0b111], 0.5, atol=1e-4)
        assert probs[1:7].sum() < 1e-4  # all others ~0

    def test_single_qubit(self):
        gates = [GateInstruction("X", (0,))]
        tensor = _build_state(1, 1, gates)
        probs = measure(tensor)
        np.testing.assert_allclose(probs[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(probs[1], 1.0, atol=1e-6)


# ============================================================
# expectation_value
# ============================================================

class TestExpectationValue:

    def test_ZZ_on_bell_state(self):
        """<Bell|ZZ|Bell> = 1.0."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZ", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev - 1.0) < 1e-5

    def test_ZI_on_bell_state(self):
        """<Bell|ZI|Bell> = 0.0."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZI", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev) < 1e-5

    def test_IZ_on_bell_state(self):
        """<Bell|IZ|Bell> = 0.0."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("IZ", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev) < 1e-5

    def test_Z_on_zero_state(self):
        """<0|Z|0> = 1.0."""
        tensor = _build_state(1, 1, [])
        h = Hamiltonian.from_pauli_list([("Z", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev - 1.0) < 1e-5

    def test_Z_on_one_state(self):
        """<1|Z|1> = -1.0."""
        tensor = _build_state(1, 1, [GateInstruction("X", (0,))])
        h = Hamiltonian.from_pauli_list([("Z", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev + 1.0) < 1e-5

    def test_X_on_plus_state(self):
        """<+|X|+> = 1.0."""
        tensor = _build_state(1, 1, [GateInstruction("H", (0,))])
        h = Hamiltonian.from_pauli_list([("X", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev - 1.0) < 1e-5

    def test_multi_term_hamiltonian(self):
        """H = ZI + 0.5*IZ on |00>. <00|ZI|00> = 1, <00|IZ|00> = 1. Total = 1.5."""
        tensor = _build_state(2, 4, [])
        h = Hamiltonian.from_pauli_list([("ZI", 1.0), ("IZ", 0.5)])
        ev = expectation_value(tensor, h)
        assert abs(ev - 1.5) < 1e-5

    def test_identity_hamiltonian(self):
        """<psi|I|psi> = 1.0 for any normalized state."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("RY", (1,), params=(0.7,)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("II", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev - 1.0) < 1e-5

    def test_matches_density_matrix(self):
        """Compare with direct density matrix computation."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("RY", (1,), params=(0.8,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZ", 0.5), ("ZI", -0.3), ("IZ", 0.2)])
        ev = expectation_value(tensor, h)

        # Reference: psi^H @ H_matrix @ psi
        amps = contract_tensor_ring(tensor).reshape(-1)
        H_mat = h.get_density_matrix()
        ev_ref = (amps.conj() @ H_mat @ amps).real.item()
        assert abs(ev - ev_ref) < 1e-5

    def test_ghz_ZZI(self):
        """<GHZ|ZZI|GHZ>. |000>: ZZI=+1, |111>: ZZI=(−1)(−1)(1)=+1. So EV=1."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
            GateInstruction("CNOT", (1, 2)),
        ]
        tensor = _build_state(3, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZI", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev - 1.0) < 1e-4

    def test_ghz_ZZZ(self):
        """<GHZ|ZZZ|GHZ> = 0 (odd parity: ZZZ|000>=+|000>, ZZZ|111>=-|111>)."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
            GateInstruction("CNOT", (1, 2)),
        ]
        tensor = _build_state(3, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZZ", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev) < 1e-4
