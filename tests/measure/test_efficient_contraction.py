"""Tests for efficient contraction measurement.

Efficient contraction computes exact expectation values via double-layer
transfer matrices. Works for both chain and ring topology.
Only supports Z/I Hamiltonians.
"""

import math
import pytest
import torch

from qiskit_trev.tensor_ring.state import TensorRingState, GateInstruction
from qiskit_trev.hamiltonian import Hamiltonian
from qiskit_trev.measure.efficient_contraction import expectation_value
from qiskit_trev.measure.full_contraction import (
    expectation_value as full_contraction_ev,
)


def _build_state(num_qubits, rank, gates):
    return TensorRingState(num_qubits=num_qubits, rank=rank).build(gates)


# ============================================================
# Basic expectation values
# ============================================================

class TestBasicExpectation:

    def test_Z_on_zero(self):
        """<0|Z|0> = 1.0."""
        tensor = _build_state(1, 1, [])
        h = Hamiltonian.from_pauli_list([("Z", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev - 1.0) < 1e-5

    def test_Z_on_one(self):
        """<1|Z|1> = -1.0."""
        tensor = _build_state(1, 1, [GateInstruction("X", (0,))])
        h = Hamiltonian.from_pauli_list([("Z", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev + 1.0) < 1e-5

    def test_Z_on_plus(self):
        """<+|Z|+> = 0.0."""
        tensor = _build_state(1, 1, [GateInstruction("H", (0,))])
        h = Hamiltonian.from_pauli_list([("Z", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev) < 1e-5

    def test_ZI_on_zero_state(self):
        """<00|ZI|00> = 1.0."""
        tensor = _build_state(2, 4, [])
        h = Hamiltonian.from_pauli_list([("ZI", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev - 1.0) < 1e-5

    def test_IZ_on_zero_state(self):
        """<00|IZ|00> = 1.0."""
        tensor = _build_state(2, 4, [])
        h = Hamiltonian.from_pauli_list([("IZ", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev - 1.0) < 1e-5

    def test_identity(self):
        """<psi|II|psi> = 1.0 for any normalized state."""
        gates = [GateInstruction("H", (0,)), GateInstruction("RY", (1,), params=(0.7,))]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("II", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev - 1.0) < 1e-5


# ============================================================
# Ring topology (entangled states)
# ============================================================

class TestRingTopology:

    def test_ZZ_on_bell_state(self):
        """<Bell|ZZ|Bell> = 1.0 (ring topology)."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZ", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev - 1.0) < 1e-4

    def test_ZI_on_bell_state(self):
        """<Bell|ZI|Bell> = 0.0."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZI", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev) < 1e-4

    def test_IZ_on_bell_state(self):
        """<Bell|IZ|Bell> = 0.0."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("IZ", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev) < 1e-4

    def test_ghz_ZZI(self):
        """<GHZ|ZZI|GHZ> = 1.0."""
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
        """<GHZ|ZZZ|GHZ> = 0 (odd parity)."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
            GateInstruction("CNOT", (1, 2)),
        ]
        tensor = _build_state(3, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZZ", 1.0)])
        ev = expectation_value(tensor, h)
        assert abs(ev) < 1e-4


# ============================================================
# Multi-term Hamiltonians
# ============================================================

class TestMultiTerm:

    def test_two_terms(self):
        """H = ZI + 0.5*IZ on |00>. Expected: 1.5."""
        tensor = _build_state(2, 4, [])
        h = Hamiltonian.from_pauli_list([("ZI", 1.0), ("IZ", 0.5)])
        ev = expectation_value(tensor, h)
        assert abs(ev - 1.5) < 1e-5

    def test_heisenberg_zz(self):
        """H = ZZII + IZZI + IIZZ on bell pair (0,1)."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(4, 4, gates)
        h = Hamiltonian.from_pauli_list([
            ("ZZII", 1.0), ("IZZI", 0.5), ("IIZZ", 0.5),
        ])
        ev = expectation_value(tensor, h)
        ev_ref = full_contraction_ev(tensor, h)
        assert abs(ev - ev_ref) < 1e-4

    def test_complex_coefficients(self):
        tensor = _build_state(2, 4, [GateInstruction("H", (0,))])
        h = Hamiltonian.from_pauli_list([("ZI", 0.3), ("IZ", -0.7), ("ZZ", 0.2)])
        ev = expectation_value(tensor, h)
        ev_ref = full_contraction_ev(tensor, h)
        assert abs(ev - ev_ref) < 1e-4


# ============================================================
# Matches full contraction
# ============================================================

class TestMatchesFullContraction:

    def test_product_state(self):
        gates = [
            GateInstruction("RY", (0,), params=(0.5,)),
            GateInstruction("RY", (1,), params=(1.0,)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZ", 0.5), ("ZI", -0.3), ("IZ", 0.2)])
        ev = expectation_value(tensor, h)
        ev_ref = full_contraction_ev(tensor, h)
        assert abs(ev - ev_ref) < 1e-5

    def test_entangled_state(self):
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("RY", (1,), params=(0.8,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZ", 0.5), ("ZI", -0.3), ("IZ", 0.2)])
        ev = expectation_value(tensor, h)
        ev_ref = full_contraction_ev(tensor, h)
        assert abs(ev - ev_ref) < 1e-4

    def test_4qubit_circuit(self):
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("RY", (1,), params=(0.8,)),
            GateInstruction("X", (2,)),
            GateInstruction("RZ", (3,), params=(1.2,)),
            GateInstruction("CNOT", (0, 1)),
            GateInstruction("CNOT", (2, 3)),
        ]
        tensor = _build_state(4, 8, gates)
        h = Hamiltonian.from_pauli_list([
            ("ZZII", 0.5), ("IZZI", -0.3), ("IIZZ", 0.2), ("ZIII", 0.1),
        ])
        ev = expectation_value(tensor, h)
        ev_ref = full_contraction_ev(tensor, h)
        assert abs(ev - ev_ref) < 1e-3
