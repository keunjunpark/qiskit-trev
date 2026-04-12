"""Cross-method consistency tests.

Verify that all measurement methods agree on the same circuit + Hamiltonian.
Exact methods (full contraction, efficient contraction) match to machine precision.
Sampling methods (right suffix) match within statistical bounds.
"""

import math
import pytest
import torch

from qiskit_trev.tensor_ring.state import TensorRingState, GateInstruction
from qiskit_trev.hamiltonian import Hamiltonian
from qiskit_trev.measure.full_contraction import expectation_value as ev_full
from qiskit_trev.measure.efficient_contraction import expectation_value as ev_efficient
from qiskit_trev.measure.right_suffix import expectation_value as ev_right_suffix


def _build_state(num_qubits, rank, gates):
    return TensorRingState(num_qubits=num_qubits, rank=rank).build(gates)


# ============================================================
# Exact methods agree (full vs efficient contraction)
# ============================================================

class TestExactMethodsAgree:

    def test_bell_ZZ(self):
        gates = [GateInstruction("H", (0,)), GateInstruction("CNOT", (0, 1))]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZ", 1.0)])
        assert abs(ev_full(tensor, h) - ev_efficient(tensor, h)) < 1e-5

    def test_bell_ZI(self):
        gates = [GateInstruction("H", (0,)), GateInstruction("CNOT", (0, 1))]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZI", 1.0)])
        assert abs(ev_full(tensor, h) - ev_efficient(tensor, h)) < 1e-5

    def test_ghz_multi_term(self):
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
            GateInstruction("CNOT", (1, 2)),
        ]
        tensor = _build_state(3, 4, gates)
        h = Hamiltonian.from_pauli_list([
            ("ZZI", 1.0), ("IZZ", 0.5), ("ZIZ", -0.3),
        ])
        assert abs(ev_full(tensor, h) - ev_efficient(tensor, h)) < 1e-4

    def test_4qubit_complex_circuit(self):
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
        assert abs(ev_full(tensor, h) - ev_efficient(tensor, h)) < 1e-3

    def test_identity_hamiltonian(self):
        gates = [GateInstruction("H", (0,)), GateInstruction("RY", (1,), params=(0.7,))]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("II", 1.0)])
        assert abs(ev_full(tensor, h) - 1.0) < 1e-5
        assert abs(ev_efficient(tensor, h) - 1.0) < 1e-5


# ============================================================
# Sampling method (right suffix) agrees with exact
# ============================================================

class TestSamplingMatchesExact:

    def test_bell_ZZ(self):
        gates = [GateInstruction("H", (0,)), GateInstruction("CNOT", (0, 1))]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZ", 1.0)])
        exact = ev_full(tensor, h)
        sampled = ev_right_suffix(tensor, h, shots=50000)
        assert abs(exact - sampled) < 0.05

    def test_ghz_multi_term(self):
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
            GateInstruction("CNOT", (1, 2)),
        ]
        tensor = _build_state(3, 4, gates)
        h = Hamiltonian.from_pauli_list([
            ("ZZI", 1.0), ("IZZ", 0.5), ("ZIZ", -0.3),
        ])
        exact = ev_full(tensor, h)
        sampled = ev_right_suffix(tensor, h, shots=100000)
        assert abs(exact - sampled) < 0.1

    def test_product_state_multi_term(self):
        gates = [
            GateInstruction("RY", (0,), params=(0.5,)),
            GateInstruction("RY", (1,), params=(1.0,)),
        ]
        tensor = _build_state(2, 4, gates)
        h = Hamiltonian.from_pauli_list([("ZZ", 0.5), ("ZI", -0.3), ("IZ", 0.2)])
        exact = ev_full(tensor, h)
        sampled = ev_right_suffix(tensor, h, shots=100000)
        assert abs(exact - sampled) < 0.1

    def test_x_hamiltonian(self):
        """X Hamiltonian: right suffix uses QWC rotation, full uses matrix."""
        tensor = _build_state(2, 4, [GateInstruction("H", (0,))])
        h = Hamiltonian.from_pauli_list([("XI", 1.0)])
        exact = ev_full(tensor, h)
        sampled = ev_right_suffix(tensor, h, shots=50000)
        assert abs(exact - sampled) < 0.05
