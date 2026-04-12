"""Tests for right suffix sampling measurement.

Right suffix uses Monte Carlo sampling with precomputed double-layer
right environments. Works for both chain and ring topology, and supports
all Pauli operators (X, Y, Z, I) via QWC grouping and basis rotations.
"""

import math
import pytest
import torch

from qiskit_trev.tensor_ring.state import TensorRingState, GateInstruction
from qiskit_trev.hamiltonian import Hamiltonian
from qiskit_trev.measure.right_suffix import expectation_value
from qiskit_trev.measure.full_contraction import (
    expectation_value as full_contraction_ev,
)


def _build_state(num_qubits, rank, gates):
    return TensorRingState(num_qubits=num_qubits, rank=rank).build(gates)


# ============================================================
# Z/I Hamiltonians (chain topology)
# ============================================================

class TestZIChain:

    def test_Z_on_zero(self):
        tensor = _build_state(1, 1, [])
        h = Hamiltonian.from_pauli_list([("Z", 1.0)])
        ev = expectation_value(tensor, h, shots=10000)
        assert abs(ev - 1.0) < 0.05

    def test_Z_on_one(self):
        tensor = _build_state(1, 1, [GateInstruction("X", (0,))])
        h = Hamiltonian.from_pauli_list([("Z", 1.0)])
        ev = expectation_value(tensor, h, shots=10000)
        assert abs(ev + 1.0) < 0.05

    def test_Z_on_plus(self):
        tensor = _build_state(1, 1, [GateInstruction("H", (0,))])
        h = Hamiltonian.from_pauli_list([("Z", 1.0)])
        ev = expectation_value(tensor, h, shots=50000)
        assert abs(ev) < 0.05


# ============================================================
# Z/I Hamiltonians (ring topology)
# ============================================================

class TestZIRing:

    def test_ZZ_on_bell_state(self):
        """<Bell|ZZ|Bell> = 1.0."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZ", 1.0)])
        ev = expectation_value(tensor, h, shots=50000)
        assert abs(ev - 1.0) < 0.05

    def test_ZI_on_bell_state(self):
        """<Bell|ZI|Bell> = 0.0."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZI", 1.0)])
        ev = expectation_value(tensor, h, shots=50000)
        assert abs(ev) < 0.05

    def test_ghz_ZZI(self):
        """<GHZ|ZZI|GHZ> = 1.0."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
            GateInstruction("CNOT", (1, 2)),
        ]
        tensor = _build_state(3, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZI", 1.0)])
        ev = expectation_value(tensor, h, shots=50000)
        assert abs(ev - 1.0) < 0.05


# ============================================================
# X/Y Hamiltonians (QWC grouping + basis rotation)
# ============================================================

class TestXYHamiltonians:

    def test_X_on_plus_state(self):
        """<+|X|+> = 1.0."""
        tensor = _build_state(1, 1, [GateInstruction("H", (0,))])
        h = Hamiltonian.from_pauli_list([("X", 1.0)])
        ev = expectation_value(tensor, h, shots=10000)
        assert abs(ev - 1.0) < 0.05

    def test_X_on_zero(self):
        """<0|X|0> = 0.0."""
        tensor = _build_state(1, 1, [])
        h = Hamiltonian.from_pauli_list([("X", 1.0)])
        ev = expectation_value(tensor, h, shots=50000)
        assert abs(ev) < 0.05

    def test_Y_on_zero(self):
        """<0|Y|0> = 0.0."""
        tensor = _build_state(1, 1, [])
        h = Hamiltonian.from_pauli_list([("Y", 1.0)])
        ev = expectation_value(tensor, h, shots=50000)
        assert abs(ev) < 0.05


# ============================================================
# Multi-term + matches full contraction
# ============================================================

class TestMatchesFullContraction:

    def test_multi_term_zi(self):
        """H = ZI + 0.5*IZ on |00>. Expected: 1.5."""
        tensor = _build_state(2, 4, [])
        h = Hamiltonian.from_pauli_list([("ZI", 1.0), ("IZ", 0.5)])
        ev = expectation_value(tensor, h, shots=50000)
        assert abs(ev - 1.5) < 0.1

    def test_entangled_state_zi(self):
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("RY", (1,), params=(0.8,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZ", 0.5), ("ZI", -0.3), ("IZ", 0.2)])
        ev = expectation_value(tensor, h, shots=100000)
        ev_ref = full_contraction_ev(tensor, h)
        assert abs(ev - ev_ref) < 0.1

    def test_mixed_pauli_hamiltonian(self):
        """Hamiltonian with X and Z terms."""
        tensor = _build_state(2, 4, [GateInstruction("H", (0,))])
        h = Hamiltonian.from_pauli_list([("ZI", 0.5), ("XI", 0.3)])
        ev = expectation_value(tensor, h, shots=100000)
        ev_ref = full_contraction_ev(tensor, h)
        assert abs(ev - ev_ref) < 0.1
