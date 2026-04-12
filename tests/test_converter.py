"""Tests for Qiskit QuantumCircuit / SparsePauliOp converter."""

import math
import pytest
import numpy as np
import torch

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

from qiskit_trev.converter import (
    circuit_to_gate_instructions,
    sparse_pauli_op_to_hamiltonian,
)
from qiskit_trev.tensor_ring.state import GateInstruction


# ============================================================
# circuit_to_gate_instructions
# ============================================================

class TestCircuitToGateInstructions:

    def test_single_h_gate(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        gates, num_params = circuit_to_gate_instructions(qc)
        assert len(gates) == 1
        assert gates[0].name == "H"
        assert gates[0].qubits == (0,)
        assert gates[0].params == ()
        assert num_params == 0

    def test_single_x_gate(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        gates, num_params = circuit_to_gate_instructions(qc)
        assert gates[0].name == "X"

    def test_single_y_gate(self):
        qc = QuantumCircuit(1)
        qc.y(0)
        gates, num_params = circuit_to_gate_instructions(qc)
        assert gates[0].name == "Y"

    def test_single_z_gate(self):
        qc = QuantumCircuit(1)
        qc.z(0)
        gates, num_params = circuit_to_gate_instructions(qc)
        assert gates[0].name == "Z"

    def test_rx_gate(self):
        qc = QuantumCircuit(1)
        qc.rx(0.5, 0)
        gates, num_params = circuit_to_gate_instructions(qc)
        assert gates[0].name == "RX"
        assert gates[0].qubits == (0,)
        assert len(gates[0].params) == 1
        assert abs(gates[0].params[0] - 0.5) < 1e-10

    def test_ry_gate(self):
        qc = QuantumCircuit(1)
        qc.ry(1.3, 0)
        gates, num_params = circuit_to_gate_instructions(qc)
        assert gates[0].name == "RY"
        assert abs(gates[0].params[0] - 1.3) < 1e-10

    def test_rz_gate(self):
        qc = QuantumCircuit(1)
        qc.rz(0.7, 0)
        gates, num_params = circuit_to_gate_instructions(qc)
        assert gates[0].name == "RZ"
        assert abs(gates[0].params[0] - 0.7) < 1e-10

    def test_cx_gate(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        gates, num_params = circuit_to_gate_instructions(qc)
        assert gates[0].name == "CNOT"
        assert gates[0].qubits == (0, 1)

    def test_swap_gate(self):
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        gates, num_params = circuit_to_gate_instructions(qc)
        assert gates[0].name == "SWAP"
        assert gates[0].qubits == (0, 1)

    def test_rzz_gate(self):
        qc = QuantumCircuit(2)
        qc.rzz(0.5, 0, 1)
        gates, num_params = circuit_to_gate_instructions(qc)
        assert gates[0].name == "ZZ"
        assert gates[0].qubits == (0, 1)
        assert abs(gates[0].params[0] - 0.5) < 1e-10

    def test_multi_gate_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(0.5, 1)
        qc.cx(0, 1)
        gates, num_params = circuit_to_gate_instructions(qc)
        assert len(gates) == 3
        assert gates[0].name == "H"
        assert gates[1].name == "RX"
        assert gates[2].name == "CNOT"

    def test_qubit_indices_preserved(self):
        qc = QuantumCircuit(3)
        qc.h(2)
        qc.cx(1, 0)
        gates, _ = circuit_to_gate_instructions(qc)
        assert gates[0].qubits == (2,)
        assert gates[1].qubits == (1, 0)

    def test_parameterized_circuit(self):
        """Parameterized circuit with symbolic Parameters."""
        theta = Parameter('theta')
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        # Bind parameters
        bound = qc.assign_parameters({theta: 0.5})
        gates, num_params = circuit_to_gate_instructions(bound)
        assert gates[0].name == "RY"
        assert abs(gates[0].params[0] - 0.5) < 1e-10

    def test_efficient_su2(self):
        """Standard library circuit parses after decomposition."""
        ansatz = EfficientSU2(4, reps=1)
        bound = ansatz.assign_parameters(np.random.randn(ansatz.num_parameters))
        decomposed = bound.decompose()
        gates, num_params = circuit_to_gate_instructions(decomposed)
        assert len(gates) > 0

    def test_u_gate(self):
        """U3/U gate."""
        qc = QuantumCircuit(1)
        qc.u(0.5, 1.0, 1.5, 0)
        gates, _ = circuit_to_gate_instructions(qc)
        assert gates[0].name == "U3"
        assert len(gates[0].params) == 3
        assert abs(gates[0].params[0] - 0.5) < 1e-10
        assert abs(gates[0].params[1] - 1.0) < 1e-10
        assert abs(gates[0].params[2] - 1.5) < 1e-10

    def test_identity_gate(self):
        qc = QuantumCircuit(1)
        qc.id(0)
        gates, _ = circuit_to_gate_instructions(qc)
        assert gates[0].name == "I"

    def test_empty_circuit(self):
        qc = QuantumCircuit(2)
        gates, num_params = circuit_to_gate_instructions(qc)
        assert len(gates) == 0
        assert num_params == 0

    def test_measurement_gates_skipped(self):
        """Measurement and barrier should be skipped."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.barrier()
        qc.measure([0, 1], [0, 1])
        gates, _ = circuit_to_gate_instructions(qc)
        assert len(gates) == 1
        assert gates[0].name == "H"


# ============================================================
# sparse_pauli_op_to_hamiltonian
# ============================================================

class TestSparsePauliOpToHamiltonian:

    def test_single_term(self):
        op = SparsePauliOp.from_list([("ZZ", 1.0)])
        h = sparse_pauli_op_to_hamiltonian(op)
        assert h.num_qubits == 2
        assert h.paulis == ["ZZ"]
        assert h.coefficients == [1.0]

    def test_multi_term(self):
        op = SparsePauliOp.from_list([("ZZII", 1.0), ("IZZI", 0.5), ("IIZZ", 0.5)])
        h = sparse_pauli_op_to_hamiltonian(op)
        assert h.num_qubits == 4
        assert len(h.paulis) == 3

    def test_coefficients(self):
        op = SparsePauliOp.from_list([("ZI", 0.3), ("IZ", -0.7)])
        h = sparse_pauli_op_to_hamiltonian(op)
        assert abs(h.coefficients[0] - 0.3) < 1e-10
        assert abs(h.coefficients[1] + 0.7) < 1e-10

    def test_all_paulis(self):
        """Qiskit LE 'IXYZ' → our BE 'ZYXI'."""
        op = SparsePauliOp.from_list([("IXYZ", 1.0)])
        h = sparse_pauli_op_to_hamiltonian(op)
        assert h.paulis == ["ZYXI"]

    def test_identity(self):
        op = SparsePauliOp.from_list([("II", 2.0)])
        h = sparse_pauli_op_to_hamiltonian(op)
        assert h.paulis == ["II"]
        assert h.coefficients == [2.0]

    def test_pauli_ordering(self):
        """Qiskit uses little-endian (qubit 0 = rightmost).
        Our Hamiltonian uses big-endian (qubit 0 = leftmost).
        Conversion should reverse the string."""
        op = SparsePauliOp.from_list([("ZI", 1.0)])  # Qiskit: Z on qubit 1, I on qubit 0
        h = sparse_pauli_op_to_hamiltonian(op)
        # Our convention: qubit 0 = leftmost, so "ZI" from Qiskit becomes "IZ" in our convention
        assert h.paulis == ["IZ"]

    def test_simplify(self):
        """Duplicate terms should be simplified."""
        op = SparsePauliOp.from_list([("ZZ", 0.5), ("ZZ", 0.5)])
        op = op.simplify()
        h = sparse_pauli_op_to_hamiltonian(op)
        assert len(h.paulis) == 1
        assert abs(h.coefficients[0] - 1.0) < 1e-10
