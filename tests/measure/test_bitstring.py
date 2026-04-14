"""Tests for bitstring extraction (greedy argmax)."""

import math
import pytest
import torch
import numpy as np

from qiskit_trev.tensor_ring.state import TensorRingState, GateInstruction
from qiskit_trev.measure.bitstring import argmax_bitstring
from qiskit_trev.measure.full_contraction import contract_tensor_ring


def _build_state(num_qubits, rank, gates):
    return TensorRingState(num_qubits=num_qubits, rank=rank).build(gates)


class TestArgmaxBitstring:

    def test_zero_state(self):
        """All-zero state → '00'."""
        tensor = _build_state(2, 4, [])
        bs = argmax_bitstring(tensor)
        assert bs == "00"

    def test_one_state(self):
        """X on both qubits → '11'."""
        tensor = _build_state(2, 4, [
            GateInstruction("X", (0,)),
            GateInstruction("X", (1,)),
        ])
        bs = argmax_bitstring(tensor)
        assert bs == "11"

    def test_single_qubit_zero(self):
        tensor = _build_state(1, 1, [])
        bs = argmax_bitstring(tensor)
        assert bs == "0"

    def test_single_qubit_one(self):
        tensor = _build_state(1, 1, [GateInstruction("X", (0,))])
        bs = argmax_bitstring(tensor)
        assert bs == "1"

    def test_x_on_qubit_0(self):
        """X(0) → |10> → '10'."""
        tensor = _build_state(2, 4, [GateInstruction("X", (0,))])
        bs = argmax_bitstring(tensor)
        assert bs == "10"

    def test_x_on_qubit_1(self):
        """X(1) → |01> → '01'."""
        tensor = _build_state(2, 4, [GateInstruction("X", (1,))])
        bs = argmax_bitstring(tensor)
        assert bs == "01"

    def test_bell_state(self):
        """Bell state → either '00' or '11' (both have equal probability)."""
        tensor = _build_state(2, 4, [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ])
        bs = argmax_bitstring(tensor)
        assert bs in ("00", "11")

    def test_ghz_3_qubit(self):
        """GHZ → either '000' or '111'."""
        tensor = _build_state(3, 4, [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
            GateInstruction("CNOT", (1, 2)),
        ])
        bs = argmax_bitstring(tensor)
        assert bs in ("000", "111")

    def test_returns_string(self):
        tensor = _build_state(2, 4, [])
        bs = argmax_bitstring(tensor)
        assert isinstance(bs, str)
        assert len(bs) == 2

    def test_4_qubit_product(self):
        """X on qubits 0 and 2 → |1010> → '1010'."""
        tensor = _build_state(4, 4, [
            GateInstruction("X", (0,)),
            GateInstruction("X", (2,)),
        ])
        bs = argmax_bitstring(tensor)
        assert bs == "1010"

    def test_matches_full_contraction_argmax(self):
        """Argmax should match the index of max probability from full contraction."""
        tensor = _build_state(3, 4, [
            GateInstruction("X", (0,)),
            GateInstruction("X", (2,)),
        ])
        bs = argmax_bitstring(tensor)

        # Full contraction reference
        amps = contract_tensor_ring(tensor).reshape(-1)
        probs = (amps * amps.conj()).real.numpy()
        max_idx = np.argmax(probs)
        expected = format(max_idx, f'0{3}b')
        assert bs == expected

    def test_biased_state(self):
        """RY(small angle) on qubit 0 should still give '0' as most probable."""
        tensor = _build_state(1, 1, [
            GateInstruction("RY", (0,), params=(0.1,)),
        ])
        bs = argmax_bitstring(tensor)
        assert bs == "0"  # cos^2(0.05) ≈ 0.9975 >> sin^2(0.05)

    def test_biased_state_large_angle(self):
        """RY(pi - 0.1) should give '1' as most probable."""
        tensor = _build_state(1, 1, [
            GateInstruction("RY", (0,), params=(math.pi - 0.1,)),
        ])
        bs = argmax_bitstring(tensor)
        assert bs == "1"


class TestArgmaxBitstringEdgeCases:

    def test_normalize_every_triggers(self):
        """normalize_every=1 triggers normalization at every step (lines 64-65)."""
        tensor = _build_state(4, 4, [
            GateInstruction("X", (0,)),
            GateInstruction("X", (2,)),
        ])
        # normalize_every=1 normalizes at i=1,2,3,...
        bs = argmax_bitstring(tensor, normalize_every=1)
        assert bs == "1010"

    def test_normalize_every_matches_default(self):
        """Result with normalize_every=1 matches default behavior."""
        tensor = _build_state(3, 4, [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
            GateInstruction("CNOT", (1, 2)),
        ])
        bs_default = argmax_bitstring(tensor)
        bs_norm = argmax_bitstring(tensor, normalize_every=1)
        assert bs_norm == bs_default

    def test_zero_probability_fallback(self):
        """When both w0 and w1 clamp to 0, site defaults to 0 (line 56)."""
        # Manually craft a tensor ring with all-zero first core so that
        # both E_i0 and E_i1 are zero, causing trace to be 0 → clamp to 0.
        tensor = _build_state(2, 4, [])
        # Zero out the first site's core so both probability branches give 0
        tensor = tensor.clone()
        tensor[0] = 0.0
        bs = argmax_bitstring(tensor)
        # Should default to '0' for the zeroed site
        assert bs[0] == "0"
