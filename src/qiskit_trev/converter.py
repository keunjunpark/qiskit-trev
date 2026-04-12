"""Convert Qiskit QuantumCircuit and SparsePauliOp to internal representations.

Ported from TREV transpile.py (real_form_autograd branch).
"""

from __future__ import annotations

from typing import List, Tuple

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .tensor_ring.state import GateInstruction
from .hamiltonian import Hamiltonian


# Gate name mapping: Qiskit name → our GateInstruction name
_QISKIT_FIXED_1Q = {
    'h': 'H',
    'x': 'X',
    'y': 'Y',
    'z': 'Z',
    'id': 'I',
    'i': 'I',
}

_QISKIT_PARAM_1Q = {
    'rx': ('RX', 1),
    'ry': ('RY', 1),
    'rz': ('RZ', 1),
}

_QISKIT_U3_NAMES = {'u', 'u3'}

_QISKIT_FIXED_2Q = {
    'cx': 'CNOT',
    'cnot': 'CNOT',
    'swap': 'SWAP',
}

_QISKIT_PARAM_2Q = {
    'rzz': ('ZZ', 1),
    'zz_swap': ('ZZ_SWAP', 1),
}

_SKIP_GATES = {'measure', 'barrier', 'reset', 'delay'}


def circuit_to_gate_instructions(
    qc: QuantumCircuit,
) -> Tuple[List[GateInstruction], int]:
    """Convert a Qiskit QuantumCircuit to a list of GateInstructions.

    The circuit should have all parameters bound (numeric values).
    Measurement, barrier, and reset gates are skipped.

    Args:
        qc: Qiskit QuantumCircuit (bound parameters).

    Returns:
        (gates, num_params) where gates is a list of GateInstruction
        and num_params is the total number of parameter values consumed.
    """
    gates: List[GateInstruction] = []
    total_params = 0

    for instruction in qc.data:
        op = instruction.operation
        name = op.name.lower()

        if name in _SKIP_GATES:
            continue

        qubits = tuple(qc.find_bit(q).index for q in instruction.qubits)

        # Fixed single-qubit gates
        if name in _QISKIT_FIXED_1Q:
            gates.append(GateInstruction(_QISKIT_FIXED_1Q[name], qubits))
            continue

        # Parameterized single-qubit gates
        if name in _QISKIT_PARAM_1Q:
            our_name, n_params = _QISKIT_PARAM_1Q[name]
            params = tuple(float(p) for p in op.params[:n_params])
            gates.append(GateInstruction(our_name, qubits, params))
            total_params += n_params
            continue

        # U3/U gate
        if name in _QISKIT_U3_NAMES:
            params = tuple(float(p) for p in op.params[:3])
            gates.append(GateInstruction('U3', qubits, params))
            total_params += 3
            continue

        # Fixed two-qubit gates
        if name in _QISKIT_FIXED_2Q:
            gates.append(GateInstruction(_QISKIT_FIXED_2Q[name], qubits))
            continue

        # Parameterized two-qubit gates
        if name in _QISKIT_PARAM_2Q:
            our_name, n_params = _QISKIT_PARAM_2Q[name]
            params = tuple(float(p) for p in op.params[:n_params])
            gates.append(GateInstruction(our_name, qubits, params))
            total_params += n_params
            continue

        raise ValueError(f"Unsupported Qiskit gate: {name}")

    return gates, total_params


def sparse_pauli_op_to_hamiltonian(op: SparsePauliOp) -> Hamiltonian:
    """Convert a Qiskit SparsePauliOp to our Hamiltonian.

    Qiskit uses little-endian ordering (qubit 0 = rightmost character).
    Our Hamiltonian uses big-endian (qubit 0 = leftmost character).
    The Pauli strings are reversed during conversion.

    Args:
        op: Qiskit SparsePauliOp.

    Returns:
        Hamiltonian instance.
    """
    num_qubits = op.num_qubits
    paulis = []
    coefficients = []

    for label, coeff in op.to_list():
        # Reverse string: Qiskit little-endian → our big-endian
        paulis.append(label[::-1])
        coefficients.append(float(coeff.real) if coeff.imag == 0 else complex(coeff))

    return Hamiltonian(
        num_qubits=num_qubits,
        paulis=paulis,
        coefficients=coefficients,
    )
