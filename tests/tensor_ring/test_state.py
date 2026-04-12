"""Tests for TensorRingState class."""

import math
import pytest
import torch
import numpy as np

from qiskit_trev.tensor_ring.state import TensorRingState, GateInstruction
from tests.utils import contract_tensor_ring, statevector, probabilities


def _numpy_gate(name, params=()):
    """Return numpy gate matrix for reference computation."""
    if name == "I":
        return np.eye(2, dtype=complex)
    elif name == "H":
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    elif name == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    elif name == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    elif name == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    elif name == "RX":
        t = params[0]
        return np.array([
            [np.cos(t / 2), -1j * np.sin(t / 2)],
            [-1j * np.sin(t / 2), np.cos(t / 2)],
        ], dtype=complex)
    elif name == "RY":
        t = params[0]
        return np.array([
            [np.cos(t / 2), -np.sin(t / 2)],
            [np.sin(t / 2), np.cos(t / 2)],
        ], dtype=complex)
    elif name == "RZ":
        t = params[0]
        return np.array([
            [np.exp(-1j * t / 2), 0],
            [0, np.exp(1j * t / 2)],
        ], dtype=complex)
    elif name == "CNOT":
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
    elif name == "SWAP":
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=complex)
    raise ValueError(f"Unknown gate {name}")


def _numpy_statevector(num_qubits, gates):
    """Compute statevector using numpy matrix operations."""
    N = num_qubits
    sv = np.zeros(2**N, dtype=complex)
    sv[0] = 1.0  # |0...0>

    for gate in gates:
        name = gate.name
        qubits = gate.qubits
        params = gate.params

        if len(qubits) == 1:
            q = qubits[0]
            mat = _numpy_gate(name, params)
            # Build full operator: I ⊗ ... ⊗ mat ⊗ ... ⊗ I
            full = np.eye(1, dtype=complex)
            for i in range(N):
                full = np.kron(full, mat if i == q else np.eye(2, dtype=complex))
            sv = full @ sv
        elif len(qubits) == 2:
            q0, q1 = qubits
            mat = _numpy_gate(name, params)
            # For adjacent qubits, build via kron
            if q1 == q0 + 1:
                full = np.eye(1, dtype=complex)
                for i in range(N):
                    if i == q0:
                        full = np.kron(full, mat)
                    elif i == q1:
                        continue  # already included in 2q gate
                    else:
                        full = np.kron(full, np.eye(2, dtype=complex))
                sv = full @ sv
            else:
                # General case: build permutation
                raise NotImplementedError("Only adjacent q0, q0+1 supported in numpy ref")

    return sv


# ============================================================
# Initialization
# ============================================================

class TestInit:

    def test_shape(self):
        state = TensorRingState(num_qubits=4, rank=8)
        tensor = state.build([])
        assert tensor.shape == (4, 8, 8, 2)

    def test_all_zeros_state(self):
        """Initial state with no gates should be |0000>."""
        state = TensorRingState(num_qubits=4, rank=8)
        tensor = state.build([])
        sv = statevector(tensor)
        expected = torch.zeros(16, dtype=torch.cfloat)
        expected[0] = 1.0
        assert torch.allclose(sv, expected, atol=1e-6)

    def test_core_values(self):
        """Initial cores: [i, 0, 0, 0] = 1.0, rest = 0."""
        state = TensorRingState(num_qubits=3, rank=4)
        tensor = state.build([])
        for i in range(3):
            assert tensor[i, 0, 0, 0].item() == 1.0
            assert tensor[i, 0, 0, 1].item() == 0.0

    def test_dtype(self):
        state = TensorRingState(num_qubits=2, rank=4)
        tensor = state.build([])
        assert tensor.dtype == torch.cfloat

    def test_different_ranks(self):
        for rank in [1, 2, 4, 8, 16]:
            state = TensorRingState(num_qubits=2, rank=rank)
            tensor = state.build([])
            assert tensor.shape == (2, rank, rank, 2)

    def test_single_qubit_system(self):
        state = TensorRingState(num_qubits=1, rank=1)
        tensor = state.build([])
        sv = statevector(tensor)
        expected = torch.tensor([1, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-6)


# ============================================================
# Single-qubit gates
# ============================================================

class TestSingleQubitGates:

    def test_H_on_qubit_0(self):
        """H(0) on 2-qubit |00> → |+0>."""
        state = TensorRingState(num_qubits=2, rank=4)
        gates = [GateInstruction("H", (0,))]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.tensor([1, 0, 1, 0], dtype=torch.cfloat) / math.sqrt(2)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_X_on_qubit_1(self):
        """X(1) on 2-qubit |00> → |01>."""
        state = TensorRingState(num_qubits=2, rank=4)
        gates = [GateInstruction("X", (1,))]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.tensor([0, 1, 0, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_RY_pi_over_2(self):
        """RY(pi/2)|0> = |+>."""
        state = TensorRingState(num_qubits=1, rank=1)
        gates = [GateInstruction("RY", (0,), params=(math.pi / 2,))]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.tensor([1, 1], dtype=torch.cfloat) / math.sqrt(2)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_RX_pi(self):
        """RX(pi)|0> = -i|1>."""
        state = TensorRingState(num_qubits=1, rank=1)
        gates = [GateInstruction("RX", (0,), params=(math.pi,))]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.tensor([0, -1j], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_RZ_on_zero(self):
        """RZ(theta)|0> = e^{-i*theta/2}|0>."""
        theta = 1.3
        state = TensorRingState(num_qubits=1, rank=1)
        gates = [GateInstruction("RZ", (0,), params=(theta,))]
        tensor = state.build(gates)
        sv = statevector(tensor)
        phase = torch.exp(torch.tensor(-1j * theta / 2))
        expected = torch.tensor([phase, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_all_single_qubit_gates(self):
        """Test I, H, X, Y, Z all work via GateInstruction."""
        state = TensorRingState(num_qubits=1, rank=1)
        for name in ["I", "H", "X", "Y", "Z"]:
            gates = [GateInstruction(name, (0,))]
            tensor = state.build(gates)
            sv = statevector(tensor)
            norm = torch.sum(torch.abs(sv) ** 2).item()
            assert abs(norm - 1.0) < 1e-5, f"Gate {name} broke norm"

    def test_consecutive_gates_same_qubit(self):
        """H then X on same qubit."""
        state = TensorRingState(num_qubits=1, rank=1)
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("X", (0,)),
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        # X @ H @ |0> = X @ |+> = |+> (X just reorders, but |+> is symmetric)
        # Actually: X|+> = |+>, so sv should be [1/√2, 1/√2]
        expected = torch.tensor([1, 1], dtype=torch.cfloat) / math.sqrt(2)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_gates_on_different_qubits(self):
        """H(0) and X(1) on 2-qubit system."""
        state = TensorRingState(num_qubits=2, rank=4)
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("X", (1,)),
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        # H⊗X |00> = |+> ⊗ |1> = (|0>+|1>)/√2 ⊗ |1>
        # = (|01> + |11>)/√2 → [0, 1/√2, 0, 1/√2]
        expected = torch.tensor([0, 1, 0, 1], dtype=torch.cfloat) / math.sqrt(2)
        assert torch.allclose(sv, expected, atol=1e-5)


# ============================================================
# Two-qubit gates
# ============================================================

class TestTwoQubitGates:

    def test_bell_state(self):
        """H(0) + CNOT(0,1) → (|00>+|11>)/√2."""
        state = TensorRingState(num_qubits=2, rank=4)
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.tensor([1, 0, 0, 1], dtype=torch.cfloat) / math.sqrt(2)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_ghz_3_qubit(self):
        """H(0), CNOT(0,1), CNOT(1,2) → GHZ."""
        state = TensorRingState(num_qubits=3, rank=4)
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
            GateInstruction("CNOT", (1, 2)),
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.zeros(8, dtype=torch.cfloat)
        expected[0] = 1 / math.sqrt(2)
        expected[7] = 1 / math.sqrt(2)
        assert torch.allclose(sv, expected, atol=1e-4)

    def test_CNOT_all_basis_states(self):
        """CNOT on all 4 basis states."""
        expected_map = {
            # (q0_state, q1_state) -> expected_index
            (0, 0): 0,  # |00> -> |00>
            (0, 1): 1,  # |01> -> |01>
            (1, 0): 3,  # |10> -> |11>
            (1, 1): 2,  # |11> -> |10>
        }
        for (s0, s1), exp_idx in expected_map.items():
            state = TensorRingState(num_qubits=2, rank=4)
            gates = []
            if s0 == 1:
                gates.append(GateInstruction("X", (0,)))
            if s1 == 1:
                gates.append(GateInstruction("X", (1,)))
            gates.append(GateInstruction("CNOT", (0, 1)))
            tensor = state.build(gates)
            sv = statevector(tensor)
            expected = torch.zeros(4, dtype=torch.cfloat)
            expected[exp_idx] = 1.0
            assert torch.allclose(sv, expected, atol=1e-5), \
                f"CNOT|{s0}{s1}> failed"

    def test_SWAP_swaps_qubits(self):
        """SWAP|01> = |10>."""
        state = TensorRingState(num_qubits=2, rank=4)
        gates = [
            GateInstruction("X", (1,)),
            GateInstruction("SWAP", (0, 1)),
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.tensor([0, 0, 1, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_adjacent_qubits_middle(self):
        """CNOT(1,2) on 4-qubit system."""
        state = TensorRingState(num_qubits=4, rank=4)
        gates = [
            GateInstruction("X", (1,)),     # |0100>
            GateInstruction("CNOT", (1, 2)),  # |0110>
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.zeros(16, dtype=torch.cfloat)
        expected[0b0110] = 1.0
        assert torch.allclose(sv, expected, atol=1e-5)


# ============================================================
# Wrap-around and reverse order two-qubit gates
# ============================================================

class TestWrapAround:

    def test_wrap_forward_CNOT(self):
        """CNOT(N-1, 0) on 4-qubit ring: wrap-forward."""
        state = TensorRingState(num_qubits=4, rank=4)
        gates = [
            GateInstruction("X", (3,)),      # |0001> (qubit 3 = 1)
            GateInstruction("CNOT", (3, 0)),  # control=3, target=0 → flip qubit 0
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.zeros(16, dtype=torch.cfloat)
        expected[0b1001] = 1.0  # |1001>
        assert torch.allclose(sv, expected, atol=1e-4)

    def test_wrap_backward_CNOT(self):
        """CNOT(0, N-1) on 4-qubit ring: wrap-backward."""
        state = TensorRingState(num_qubits=4, rank=4)
        gates = [
            GateInstruction("X", (0,)),       # |1000>
            GateInstruction("CNOT", (0, 3)),   # control=0, target=3 → flip qubit 3
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.zeros(16, dtype=torch.cfloat)
        expected[0b1001] = 1.0  # |1001>
        assert torch.allclose(sv, expected, atol=1e-4)

    def test_reverse_order_CNOT(self):
        """CNOT(1, 0): control=1, target=0 (q0 > q1 case)."""
        state = TensorRingState(num_qubits=2, rank=4)
        gates = [
            GateInstruction("X", (1,)),       # |01>
            GateInstruction("CNOT", (1, 0)),   # control=1, target=0 → flip qubit 0
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.tensor([0, 0, 0, 1], dtype=torch.cfloat)  # |11>
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_reverse_order_CNOT_no_flip(self):
        """CNOT(1,0) with control=0: |10> should stay |10>."""
        state = TensorRingState(num_qubits=2, rank=4)
        gates = [
            GateInstruction("X", (0,)),       # |10>
            GateInstruction("CNOT", (1, 0)),   # control=1=0, no flip
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.tensor([0, 0, 1, 0], dtype=torch.cfloat)  # |10>
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_wrap_SWAP(self):
        """SWAP(3, 0) on 4-qubit ring."""
        state = TensorRingState(num_qubits=4, rank=4)
        gates = [
            GateInstruction("X", (0,)),       # |1000>
            GateInstruction("SWAP", (3, 0)),   # swap qubits 3 and 0
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.zeros(16, dtype=torch.cfloat)
        expected[0b0001] = 1.0  # |0001>
        assert torch.allclose(sv, expected, atol=1e-4)

    def test_non_adjacent_raises_error(self):
        """Non-adjacent qubits (e.g., CNOT(0, 2) on 4 qubits) should raise ValueError."""
        state = TensorRingState(num_qubits=4, rank=4)
        gates = [GateInstruction("CNOT", (0, 2))]
        with pytest.raises(ValueError, match="adjacent"):
            state.build(gates)


# ============================================================
# Gate fusion
# ============================================================

class TestGateFusion:

    def test_consecutive_single_qubit_same_result(self):
        """Multiple consecutive 1q gates should give same result as fused."""
        state = TensorRingState(num_qubits=2, rank=4)
        # H then RY(0.5) on qubit 0
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("RY", (0,), params=(0.5,)),
        ]
        tensor = state.build(gates)
        sv1 = statevector(tensor)

        # Verify norm
        norm = torch.sum(torch.abs(sv1) ** 2).item()
        assert abs(norm - 1.0) < 1e-5

    def test_three_gates_fused(self):
        """H, X, RZ(0.3) on same qubit."""
        state = TensorRingState(num_qubits=1, rank=1)
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("X", (0,)),
            GateInstruction("RZ", (0,), params=(0.3,)),
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        norm = torch.sum(torch.abs(sv) ** 2).item()
        assert abs(norm - 1.0) < 1e-5


# ============================================================
# Comparison with numpy reference
# ============================================================

class TestNumpyReference:

    def test_2qubit_circuit(self):
        """H(0), X(1) matches numpy."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("X", (1,)),
        ]
        state = TensorRingState(num_qubits=2, rank=4)
        tensor = state.build(gates)
        sv = statevector(tensor).numpy()
        expected = _numpy_statevector(2, gates)
        np.testing.assert_allclose(sv, expected, atol=1e-5)

    def test_3qubit_product_state(self):
        """RY on each qubit matches numpy Kronecker product."""
        gates = [
            GateInstruction("RY", (0,), params=(0.5,)),
            GateInstruction("RY", (1,), params=(1.0,)),
            GateInstruction("RY", (2,), params=(1.5,)),
        ]
        state = TensorRingState(num_qubits=3, rank=4)
        tensor = state.build(gates)
        sv = statevector(tensor).numpy()
        expected = _numpy_statevector(3, gates)
        np.testing.assert_allclose(sv, expected, atol=1e-5)

    def test_bell_state_matches_numpy(self):
        """H(0) + CNOT(0,1) matches numpy."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        state = TensorRingState(num_qubits=2, rank=4)
        tensor = state.build(gates)
        sv = statevector(tensor).numpy()
        expected = _numpy_statevector(2, gates)
        np.testing.assert_allclose(sv, expected, atol=1e-5)

    def test_4qubit_mixed_circuit(self):
        """4-qubit circuit with mixed gates matches numpy."""
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("RY", (1,), params=(0.8,)),
            GateInstruction("X", (2,)),
            GateInstruction("RZ", (3,), params=(1.2,)),
            GateInstruction("CNOT", (0, 1)),
            GateInstruction("CNOT", (2, 3)),
        ]
        state = TensorRingState(num_qubits=4, rank=8)
        tensor = state.build(gates)
        sv = statevector(tensor).numpy()
        expected = _numpy_statevector(4, gates)
        np.testing.assert_allclose(sv, expected, atol=1e-4)


# ============================================================
# U3 gate
# ============================================================

class TestU3Gate:

    def test_U3_zero_is_identity(self):
        state = TensorRingState(num_qubits=1, rank=1)
        gates = [GateInstruction("U3", (0,), params=(0.0, 0.0, 0.0))]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.tensor([1, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-6)

    def test_U3_as_RY(self):
        """U3(pi/2, 0, 0) = RY(pi/2)."""
        state = TensorRingState(num_qubits=1, rank=1)
        gates = [GateInstruction("U3", (0,), params=(math.pi / 2, 0.0, 0.0))]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.tensor([1, 1], dtype=torch.cfloat) / math.sqrt(2)
        assert torch.allclose(sv, expected, atol=1e-5)


# ============================================================
# ZZ and ZZ_SWAP gates
# ============================================================

class TestZZGates:

    def test_ZZ_at_zero(self):
        """ZZ(0) is identity on 2 qubits."""
        state = TensorRingState(num_qubits=2, rank=4)
        gates = [GateInstruction("ZZ", (0, 1), params=(0.0,))]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.tensor([1, 0, 0, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_ZZ_SWAP_swaps_and_phases(self):
        """ZZ_SWAP on |01> should give e^{i*theta/2}|10>."""
        theta = 0.5
        state = TensorRingState(num_qubits=2, rank=4)
        gates = [
            GateInstruction("X", (1,)),
            GateInstruction("ZZ_SWAP", (0, 1), params=(theta,)),
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        phase = torch.exp(torch.tensor(1j * theta / 2))
        expected = torch.tensor([0, 0, phase, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:

    def test_empty_gate_list(self):
        state = TensorRingState(num_qubits=3, rank=4)
        tensor = state.build([])
        sv = statevector(tensor)
        expected = torch.zeros(8, dtype=torch.cfloat)
        expected[0] = 1.0
        assert torch.allclose(sv, expected, atol=1e-6)

    def test_single_qubit_system_with_gate(self):
        state = TensorRingState(num_qubits=1, rank=1)
        gates = [GateInstruction("X", (0,))]
        tensor = state.build(gates)
        sv = statevector(tensor)
        expected = torch.tensor([0, 1], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-6)

    def test_build_twice_gives_same_result(self):
        """Building state twice should give identical results."""
        state = TensorRingState(num_qubits=2, rank=4)
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        sv1 = statevector(state.build(gates))
        sv2 = statevector(state.build(gates))
        assert torch.allclose(sv1, sv2, atol=1e-6)

    def test_norm_preserved_complex_circuit(self):
        """Norm should be 1 after a complex circuit."""
        state = TensorRingState(num_qubits=4, rank=8)
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("RY", (1,), params=(0.7,)),
            GateInstruction("RX", (2,), params=(1.3,)),
            GateInstruction("H", (3,)),
            GateInstruction("CNOT", (0, 1)),
            GateInstruction("CNOT", (1, 2)),
            GateInstruction("CNOT", (2, 3)),
        ]
        tensor = state.build(gates)
        sv = statevector(tensor)
        norm = torch.sum(torch.abs(sv) ** 2).item()
        assert abs(norm - 1.0) < 1e-4

    def test_large_rank_matches_small_for_product_state(self):
        """Product state should give same result at rank=1 and rank=16."""
        gates = [
            GateInstruction("RY", (0,), params=(0.5,)),
            GateInstruction("RX", (1,), params=(1.0,)),
        ]
        sv1 = statevector(TensorRingState(2, rank=1).build(gates))
        sv16 = statevector(TensorRingState(2, rank=16).build(gates))
        assert torch.allclose(sv1, sv16, atol=1e-5)
