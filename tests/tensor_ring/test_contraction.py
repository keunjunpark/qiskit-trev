"""Tests for tensor ring gate contraction functions."""

import math
import pytest
import torch
import numpy as np

from qiskit_trev.tensor_ring.gates import H, X, Y, Z, RX, RY, RZ, CNOT, SWAP, I
from qiskit_trev.tensor_ring.contraction import (
    apply_single_qubit_gate,
    apply_double_qubit_gate,
    swap_gate_matrix,
)
from tests.utils import contract_tensor_ring, statevector, probabilities


def _make_zero_state(num_qubits, rank):
    """Create |0...0> tensor ring state."""
    tensor = torch.zeros((num_qubits, rank, rank, 2), dtype=torch.cfloat)
    tensor[:, 0, 0, 0] = 1.0
    return tensor


# ============================================================
# Single-qubit gate application
# ============================================================

class TestApplySingleQubitGate:

    def test_output_shape(self):
        core = torch.randn(4, 4, 2, dtype=torch.cfloat)
        result = apply_single_qubit_gate(H(), core)
        assert result.shape == (4, 4, 2)

    def test_H_on_zero_single_qubit(self):
        """H|0> = |+> on a 1-qubit tensor ring."""
        tensor = _make_zero_state(1, 1)
        tensor[0] = apply_single_qubit_gate(H(), tensor[0])
        sv = statevector(tensor)
        expected = torch.tensor([1, 1], dtype=torch.cfloat) / math.sqrt(2)
        assert torch.allclose(sv, expected, atol=1e-6)

    def test_X_on_zero_single_qubit(self):
        """X|0> = |1>."""
        tensor = _make_zero_state(1, 1)
        tensor[0] = apply_single_qubit_gate(X(), tensor[0])
        sv = statevector(tensor)
        expected = torch.tensor([0, 1], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-6)

    def test_Y_on_zero_single_qubit(self):
        """Y|0> = i|1>."""
        tensor = _make_zero_state(1, 1)
        tensor[0] = apply_single_qubit_gate(Y(), tensor[0])
        sv = statevector(tensor)
        expected = torch.tensor([0, 1j], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-6)

    def test_Z_on_zero_leaves_unchanged(self):
        """Z|0> = |0>."""
        tensor = _make_zero_state(1, 1)
        tensor[0] = apply_single_qubit_gate(Z(), tensor[0])
        sv = statevector(tensor)
        expected = torch.tensor([1, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-6)

    def test_identity_leaves_state_unchanged(self):
        tensor = _make_zero_state(1, 1)
        tensor[0] = apply_single_qubit_gate(I(), tensor[0])
        sv = statevector(tensor)
        expected = torch.tensor([1, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-6)

    def test_preserves_norm(self):
        """Applying a unitary gate preserves the norm."""
        tensor = _make_zero_state(1, 4)
        tensor[0] = apply_single_qubit_gate(RY(0.7), tensor[0])
        sv = statevector(tensor)
        norm = torch.sum(torch.abs(sv) ** 2).item()
        assert abs(norm - 1.0) < 1e-5

    def test_RY_pi_over_2_on_zero(self):
        """RY(pi/2)|0> = |+>."""
        tensor = _make_zero_state(1, 1)
        tensor[0] = apply_single_qubit_gate(RY(math.pi / 2), tensor[0])
        sv = statevector(tensor)
        expected = torch.tensor([1, 1], dtype=torch.cfloat) / math.sqrt(2)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_two_consecutive_gates_H_then_H(self):
        """HH|0> = |0>."""
        tensor = _make_zero_state(1, 1)
        tensor[0] = apply_single_qubit_gate(H(), tensor[0])
        tensor[0] = apply_single_qubit_gate(H(), tensor[0])
        sv = statevector(tensor)
        expected = torch.tensor([1, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_gate_on_specific_qubit_in_multi_qubit(self):
        """Apply H to qubit 0 of 2-qubit |00>, get |+0>."""
        tensor = _make_zero_state(2, 4)
        tensor[0] = apply_single_qubit_gate(H(), tensor[0])
        sv = statevector(tensor)
        # |+0> = (|00> + |10>) / sqrt(2) → amplitudes [1/√2, 0, 1/√2, 0]
        expected = torch.tensor([1, 0, 1, 0], dtype=torch.cfloat) / math.sqrt(2)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_gate_on_qubit_1_of_2(self):
        """Apply X to qubit 1 of 2-qubit |00>, get |01>."""
        tensor = _make_zero_state(2, 4)
        tensor[1] = apply_single_qubit_gate(X(), tensor[1])
        sv = statevector(tensor)
        expected = torch.tensor([0, 1, 0, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_dtype_preserved(self):
        core = torch.zeros(2, 2, 2, dtype=torch.cfloat)
        core[0, 0, 0] = 1.0
        result = apply_single_qubit_gate(H(), core)
        assert result.dtype == torch.cfloat


# ============================================================
# Two-qubit gate application
# ============================================================

class TestApplyDoubleQubitGate:

    def test_output_shapes(self):
        core_a = torch.randn(4, 4, 2, dtype=torch.cfloat)
        core_b = torch.randn(4, 4, 2, dtype=torch.cfloat)
        new_a, new_b = apply_double_qubit_gate(CNOT(), core_a, core_b)
        assert new_a.shape[2] == 2
        assert new_b.shape[2] == 2
        # Bond dimensions may change but physical dim stays 2
        assert new_a.shape[0] == core_a.shape[0]  # left bond preserved
        assert new_b.shape[1] == core_b.shape[1]  # right bond preserved

    def test_CNOT_on_00(self):
        """CNOT|00> = |00>."""
        tensor = _make_zero_state(2, 4)
        tensor[0], tensor[1] = apply_double_qubit_gate(CNOT(), tensor[0], tensor[1])
        sv = statevector(tensor)
        expected = torch.tensor([1, 0, 0, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_CNOT_on_10(self):
        """CNOT|10> = |11>."""
        tensor = _make_zero_state(2, 4)
        tensor[0] = apply_single_qubit_gate(X(), tensor[0])
        tensor[0], tensor[1] = apply_double_qubit_gate(CNOT(), tensor[0], tensor[1])
        sv = statevector(tensor)
        expected = torch.tensor([0, 0, 0, 1], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_CNOT_on_01(self):
        """CNOT|01> = |01>."""
        tensor = _make_zero_state(2, 4)
        tensor[1] = apply_single_qubit_gate(X(), tensor[1])
        tensor[0], tensor[1] = apply_double_qubit_gate(CNOT(), tensor[0], tensor[1])
        sv = statevector(tensor)
        expected = torch.tensor([0, 1, 0, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_CNOT_on_11(self):
        """CNOT|11> = |10>."""
        tensor = _make_zero_state(2, 4)
        tensor[0] = apply_single_qubit_gate(X(), tensor[0])
        tensor[1] = apply_single_qubit_gate(X(), tensor[1])
        tensor[0], tensor[1] = apply_double_qubit_gate(CNOT(), tensor[0], tensor[1])
        sv = statevector(tensor)
        expected = torch.tensor([0, 0, 1, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_bell_state(self):
        """H(0) + CNOT(0,1) creates Bell state (|00>+|11>)/sqrt(2)."""
        tensor = _make_zero_state(2, 4)
        tensor[0] = apply_single_qubit_gate(H(), tensor[0])
        tensor[0], tensor[1] = apply_double_qubit_gate(CNOT(), tensor[0], tensor[1])
        sv = statevector(tensor)
        expected = torch.tensor([1, 0, 0, 1], dtype=torch.cfloat) / math.sqrt(2)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_SWAP_on_01(self):
        """SWAP|01> = |10>."""
        tensor = _make_zero_state(2, 4)
        tensor[1] = apply_single_qubit_gate(X(), tensor[1])
        tensor[0], tensor[1] = apply_double_qubit_gate(SWAP(), tensor[0], tensor[1])
        sv = statevector(tensor)
        expected = torch.tensor([0, 0, 1, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_SWAP_on_10(self):
        """SWAP|10> = |01>."""
        tensor = _make_zero_state(2, 4)
        tensor[0] = apply_single_qubit_gate(X(), tensor[0])
        tensor[0], tensor[1] = apply_double_qubit_gate(SWAP(), tensor[0], tensor[1])
        sv = statevector(tensor)
        expected = torch.tensor([0, 1, 0, 0], dtype=torch.cfloat)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_rank_truncation_product_state(self):
        """Product state |11> needs rank=1. Truncation should be exact."""
        tensor = _make_zero_state(2, 1)
        tensor[0] = apply_single_qubit_gate(X(), tensor[0])
        tensor[0], tensor[1] = apply_double_qubit_gate(
            CNOT(), tensor[0], tensor[1], max_rank=1
        )
        # |11> is a product state, rank=1 should be exact
        sv = statevector(tensor)
        prob = (torch.abs(sv) ** 2).numpy()
        # Should have all weight on |11>
        assert prob[3] > 0.99

    def test_bell_state_needs_rank_2(self):
        """Bell state needs rank >= 2 for full fidelity."""
        # rank=2: should be exact
        tensor = _make_zero_state(2, 2)
        tensor[0] = apply_single_qubit_gate(H(), tensor[0])
        tensor[0], tensor[1] = apply_double_qubit_gate(
            CNOT(), tensor[0], tensor[1], max_rank=2
        )
        sv = statevector(tensor)
        expected = torch.tensor([1, 0, 0, 1], dtype=torch.cfloat) / math.sqrt(2)
        assert torch.allclose(sv, expected, atol=1e-5)

    def test_bell_state_rank_1_degrades(self):
        """Bell state at rank=1 should lose fidelity."""
        tensor = _make_zero_state(2, 1)
        tensor[0] = apply_single_qubit_gate(H(), tensor[0])
        tensor[0], tensor[1] = apply_double_qubit_gate(
            CNOT(), tensor[0], tensor[1], max_rank=1
        )
        sv = statevector(tensor)
        expected = torch.tensor([1, 0, 0, 1], dtype=torch.cfloat) / math.sqrt(2)
        fidelity = torch.abs(torch.dot(sv.conj(), expected)) ** 2
        # Fidelity should be strictly less than 1 at rank=1
        assert fidelity.item() < 0.99

    def test_preserves_norm_after_CNOT(self):
        tensor = _make_zero_state(2, 4)
        tensor[0] = apply_single_qubit_gate(H(), tensor[0])
        tensor[0], tensor[1] = apply_double_qubit_gate(CNOT(), tensor[0], tensor[1])
        sv = statevector(tensor)
        norm = torch.sum(torch.abs(sv) ** 2).item()
        assert abs(norm - 1.0) < 1e-5

    def test_preserves_norm_after_SWAP(self):
        tensor = _make_zero_state(2, 4)
        tensor[0] = apply_single_qubit_gate(RY(0.7), tensor[0])
        tensor[1] = apply_single_qubit_gate(RX(1.3), tensor[1])
        tensor[0], tensor[1] = apply_double_qubit_gate(SWAP(), tensor[0], tensor[1])
        sv = statevector(tensor)
        norm = torch.sum(torch.abs(sv) ** 2).item()
        assert abs(norm - 1.0) < 1e-4

    def test_double_CNOT_is_identity(self):
        """Applying CNOT twice should return to original state."""
        tensor = _make_zero_state(2, 4)
        tensor[0] = apply_single_qubit_gate(RY(0.8), tensor[0])
        tensor[1] = apply_single_qubit_gate(RX(0.3), tensor[1])
        sv_before = statevector(tensor.clone())

        tensor[0], tensor[1] = apply_double_qubit_gate(CNOT(), tensor[0], tensor[1])
        tensor[0], tensor[1] = apply_double_qubit_gate(CNOT(), tensor[0], tensor[1])
        sv_after = statevector(tensor)
        assert torch.allclose(sv_after, sv_before, atol=1e-4)


# ============================================================
# swap_gate_matrix
# ============================================================

class TestSwapGateMatrix:

    def test_swap_cnot(self):
        """SWAP @ CNOT @ SWAP gives reverse-CNOT."""
        swapped = swap_gate_matrix(CNOT())
        # Reverse CNOT: target=0, control=1
        # |00>->|00>, |01>->|11>, |10>->|10>, |11>->|01>
        expected = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ], dtype=torch.cfloat)
        assert torch.allclose(swapped, expected, atol=1e-6)

    def test_swap_swap_is_swap(self):
        """SWAP @ SWAP @ SWAP = SWAP (SWAP is self-inverse, conjugation gives itself)."""
        swapped = swap_gate_matrix(SWAP())
        assert torch.allclose(swapped, SWAP(), atol=1e-6)

    def test_swap_identity_is_identity(self):
        """SWAP @ I @ SWAP = I."""
        eye4 = torch.eye(4, dtype=torch.cfloat)
        swapped = swap_gate_matrix(eye4)
        assert torch.allclose(swapped, eye4, atol=1e-6)

    def test_swap_preserves_unitarity(self):
        """Swapping a unitary gate should remain unitary."""
        swapped = swap_gate_matrix(CNOT())
        identity = torch.eye(4, dtype=torch.cfloat)
        product = swapped @ swapped.conj().T
        assert torch.allclose(product, identity, atol=1e-5)

    def test_double_swap_is_original(self):
        """swap_gate_matrix(swap_gate_matrix(M)) = M."""
        original = CNOT()
        double_swapped = swap_gate_matrix(swap_gate_matrix(original))
        assert torch.allclose(double_swapped, original, atol=1e-6)


# ============================================================
# Three-qubit tests using contraction primitives
# ============================================================

class TestThreeQubitContraction:

    def test_ghz_state(self):
        """H(0), CNOT(0,1), CNOT(1,2) → GHZ state (|000>+|111>)/sqrt(2)."""
        tensor = _make_zero_state(3, 4)
        tensor[0] = apply_single_qubit_gate(H(), tensor[0])
        tensor[0], tensor[1] = apply_double_qubit_gate(CNOT(), tensor[0], tensor[1])
        tensor[1], tensor[2] = apply_double_qubit_gate(CNOT(), tensor[1], tensor[2])
        sv = statevector(tensor)
        expected = torch.zeros(8, dtype=torch.cfloat)
        expected[0] = 1 / math.sqrt(2)  # |000>
        expected[7] = 1 / math.sqrt(2)  # |111>
        assert torch.allclose(sv, expected, atol=1e-4)

    def test_3qubit_product_state(self):
        """RY on each qubit creates a product state."""
        tensor = _make_zero_state(3, 4)
        tensor[0] = apply_single_qubit_gate(RY(0.5), tensor[0])
        tensor[1] = apply_single_qubit_gate(RY(1.0), tensor[1])
        tensor[2] = apply_single_qubit_gate(RY(1.5), tensor[2])
        sv = statevector(tensor)
        norm = torch.sum(torch.abs(sv) ** 2).item()
        assert abs(norm - 1.0) < 1e-5

    def test_3qubit_product_state_matches_numpy(self):
        """Compare product state with numpy Kronecker product."""
        theta0, theta1, theta2 = 0.5, 1.0, 1.5
        tensor = _make_zero_state(3, 4)
        tensor[0] = apply_single_qubit_gate(RY(theta0), tensor[0])
        tensor[1] = apply_single_qubit_gate(RY(theta1), tensor[1])
        tensor[2] = apply_single_qubit_gate(RY(theta2), tensor[2])
        sv = statevector(tensor).numpy()

        # Numpy reference
        ket0 = np.array([1, 0], dtype=complex)
        ry = lambda t: np.array([
            [np.cos(t / 2), -np.sin(t / 2)],
            [np.sin(t / 2), np.cos(t / 2)],
        ])
        s0 = ry(theta0) @ ket0
        s1 = ry(theta1) @ ket0
        s2 = ry(theta2) @ ket0
        expected = np.kron(np.kron(s0, s1), s2)
        np.testing.assert_allclose(sv, expected, atol=1e-5)
