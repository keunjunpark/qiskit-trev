"""Tests for perfect sampling measurement.

Perfect sampling uses sequential left-to-right collapse which is correct
for chain (open boundary) topology. Tests here use product states and
rank-1 states where the ring closure is trivial.
"""

import math
import pytest
import torch
import numpy as np

from qiskit_trev.tensor_ring.state import TensorRingState, GateInstruction
from qiskit_trev.hamiltonian import Hamiltonian
from qiskit_trev.measure.perfect_sampling import measure, expectation_value


def _build_state(num_qubits, rank, gates):
    state = TensorRingState(num_qubits=num_qubits, rank=rank)
    return state.build(gates)


# ============================================================
# measure (chain-compatible states)
# ============================================================

class TestMeasure:

    def test_zero_state_all_zeros(self):
        """Sampling |00> should always give index 0."""
        tensor = _build_state(2, 4, [])
        probs = np.array(measure(tensor, shots=1000))
        assert probs[0] > 0.99

    def test_one_state(self):
        """Sampling |11> should always give index 3 (LE) or similar."""
        gates = [GateInstruction("X", (0,)), GateInstruction("X", (1,))]
        tensor = _build_state(2, 4, gates)
        probs = np.array(measure(tensor, shots=1000))
        # X|0> at both sites → all shots at one index
        assert max(probs) > 0.99

    def test_plus_state_single_qubit(self):
        """H|0> = |+> should sample ~50/50."""
        tensor = _build_state(1, 1, [GateInstruction("H", (0,))])
        probs = np.array(measure(tensor, shots=10000))
        assert probs[0] > 0.4 and probs[0] < 0.6
        assert probs[1] > 0.4 and probs[1] < 0.6

    def test_product_state_2qubit(self):
        """RY on each qubit: product state, rank-1 ring closure trivial."""
        tensor = _build_state(2, 1, [
            GateInstruction("RY", (0,), params=(math.pi / 3,)),
            GateInstruction("RY", (1,), params=(math.pi / 4,)),
        ])
        probs = np.array(measure(tensor, shots=20000))
        # RY(pi/3)|0>: P(0) = cos^2(pi/6) = 3/4, P(1) = 1/4
        # RY(pi/4)|0>: P(0) = cos^2(pi/8) ≈ 0.854, P(1) ≈ 0.146
        p0_q0 = math.cos(math.pi / 6) ** 2
        p1_q0 = 1 - p0_q0
        p0_q1 = math.cos(math.pi / 8) ** 2
        p1_q1 = 1 - p0_q1
        # |00> ~ p0_q0 * p0_q1, etc.
        assert abs(probs[0] - p0_q0 * p0_q1) < 0.05  # depends on bit ordering

    def test_probabilities_sum_to_one(self):
        tensor = _build_state(2, 1, [
            GateInstruction("H", (0,)),
            GateInstruction("RY", (1,), params=(0.7,)),
        ])
        probs = np.array(measure(tensor, shots=5000))
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)

    def test_deterministic_state_all_zeros(self):
        """All-zero state: every shot should give |00...0>."""
        tensor = _build_state(3, 4, [])
        probs = np.array(measure(tensor, shots=100))
        np.testing.assert_allclose(probs[0], 1.0, atol=1e-10)


# ============================================================
# expectation_value (chain-compatible Z/I Hamiltonians)
# ============================================================

class TestExpectationValue:

    def test_Z_on_zero(self):
        """<0|Z|0> = 1.0."""
        tensor = _build_state(1, 1, [])
        h = Hamiltonian.from_pauli_list([("Z", 1.0)])
        ev = expectation_value(tensor, h, shots=10000)
        assert abs(ev - 1.0) < 0.05

    def test_Z_on_one(self):
        """<1|Z|1> = -1.0."""
        tensor = _build_state(1, 1, [GateInstruction("X", (0,))])
        h = Hamiltonian.from_pauli_list([("Z", 1.0)])
        ev = expectation_value(tensor, h, shots=10000)
        assert abs(ev + 1.0) < 0.05

    def test_Z_on_plus(self):
        """<+|Z|+> = 0.0."""
        tensor = _build_state(1, 1, [GateInstruction("H", (0,))])
        h = Hamiltonian.from_pauli_list([("Z", 1.0)])
        ev = expectation_value(tensor, h, shots=50000)
        assert abs(ev) < 0.05

    def test_ZI_on_product_state(self):
        """<0,0|ZI|0,0> = 1.0 (product state at rank=1)."""
        tensor = _build_state(2, 1, [])
        h = Hamiltonian.from_pauli_list([("ZI", 1.0)])
        ev = expectation_value(tensor, h, shots=10000)
        assert abs(ev - 1.0) < 0.05

    def test_multi_term_product_state(self):
        """H = ZI + 0.5*IZ on |00> (rank=1). Expected: 1.5."""
        tensor = _build_state(2, 1, [])
        h = Hamiltonian.from_pauli_list([("ZI", 1.0), ("IZ", 0.5)])
        ev = expectation_value(tensor, h, shots=50000)
        assert abs(ev - 1.5) < 0.05

    def test_ZZ_on_product_00(self):
        """<00|ZZ|00> = 1.0."""
        tensor = _build_state(2, 1, [])
        h = Hamiltonian.from_pauli_list([("ZZ", 1.0)])
        ev = expectation_value(tensor, h, shots=10000)
        assert abs(ev - 1.0) < 0.05

    def test_ZZ_on_product_01(self):
        """<01|ZZ|01> = -1.0."""
        tensor = _build_state(2, 1, [GateInstruction("X", (1,))])
        h = Hamiltonian.from_pauli_list([("ZZ", 1.0)])
        ev = expectation_value(tensor, h, shots=10000)
        assert abs(ev + 1.0) < 0.05

    def test_3qubit_product_state(self):
        """Product state with RY gates at rank=1."""
        tensor = _build_state(3, 1, [
            GateInstruction("RY", (0,), params=(0.5,)),
            GateInstruction("RY", (1,), params=(1.0,)),
            GateInstruction("RY", (2,), params=(1.5,)),
        ])
        # Z expectation on each qubit: <Z> = cos(theta)
        h = Hamiltonian.from_pauli_list([("ZII", 1.0)])
        ev = expectation_value(tensor, h, shots=50000)
        assert abs(ev - math.cos(0.5)) < 0.05
