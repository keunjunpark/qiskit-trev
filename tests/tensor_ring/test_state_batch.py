"""Tests for TensorRingState.build_batch."""

import math
import pytest
import torch
import numpy as np

from qiskit_trev.tensor_ring.state import TensorRingState, GateInstruction
from tests.utils import statevector


class TestBuildBatch:

    def test_shape(self):
        state = TensorRingState(num_qubits=2, rank=4)
        gates = [GateInstruction("H", (0,))]
        params = torch.zeros(3, 0)  # 3 batch, 0 params
        tensor = state.build_batch(gates, params)
        assert tensor.shape == (3, 2, 4, 4, 2)

    def test_all_zeros_batch(self):
        """Batch of zero states."""
        state = TensorRingState(num_qubits=2, rank=4)
        tensor = state.build_batch([], torch.zeros(2, 0))
        for b in range(2):
            sv = statevector(tensor[b])
            expected = torch.zeros(4, dtype=torch.cfloat)
            expected[0] = 1.0
            assert torch.allclose(sv, expected, atol=1e-6)

    def test_matches_single_build(self):
        """Batch build should match individual builds."""
        state = TensorRingState(num_qubits=2, rank=4)
        gates = [
            GateInstruction("RY", (0,), params=(0.0,)),
            GateInstruction("RX", (1,), params=(0.0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        params = torch.tensor([
            [0.5, 1.0],
            [1.0, 0.5],
            [0.0, 0.0],
        ])

        batch_tensor = state.build_batch(gates, params)

        for b in range(3):
            gates_single = [
                GateInstruction("RY", (0,), params=(params[b, 0].item(),)),
                GateInstruction("RX", (1,), params=(params[b, 1].item(),)),
                GateInstruction("CNOT", (0, 1)),
            ]
            single_tensor = state.build(gates_single)
            sv_batch = statevector(batch_tensor[b])
            sv_single = statevector(single_tensor)
            assert torch.allclose(sv_batch, sv_single, atol=1e-4), f"Batch {b} mismatch"

    def test_different_params(self):
        """Different parameter values produce different states."""
        state = TensorRingState(num_qubits=1, rank=1)
        gates = [GateInstruction("RY", (0,), params=(0.0,))]
        params = torch.tensor([[0.0], [math.pi]])
        tensor = state.build_batch(gates, params)
        sv0 = statevector(tensor[0])
        sv1 = statevector(tensor[1])
        # theta=0 → |0>, theta=pi → |1>
        assert torch.abs(sv0[0]).item() > 0.99
        assert torch.abs(sv1[1]).item() > 0.99

    def test_bell_state_batch(self):
        """Batch of bell states (no params, same for all)."""
        state = TensorRingState(num_qubits=2, rank=4)
        gates = [
            GateInstruction("H", (0,)),
            GateInstruction("CNOT", (0, 1)),
        ]
        tensor = state.build_batch(gates, torch.zeros(3, 0))
        for b in range(3):
            sv = statevector(tensor[b])
            expected = torch.tensor([1, 0, 0, 1], dtype=torch.cfloat) / math.sqrt(2)
            assert torch.allclose(sv, expected, atol=1e-4)

    def test_batch_size_1(self):
        state = TensorRingState(num_qubits=1, rank=1)
        gates = [GateInstruction("RY", (0,), params=(0.0,))]
        tensor = state.build_batch(gates, torch.tensor([[0.5]]))
        assert tensor.shape == (1, 1, 1, 1, 2)

    def test_wrap_around_2q_gate_batch(self):
        """CNOT(N-1, 0) in batch mode."""
        state = TensorRingState(num_qubits=4, rank=4)
        gates = [
            GateInstruction("X", (3,)),
            GateInstruction("CNOT", (3, 0)),
        ]
        tensor = state.build_batch(gates, torch.zeros(2, 0))
        for b in range(2):
            sv = statevector(tensor[b])
            expected = torch.zeros(16, dtype=torch.cfloat)
            expected[0b1001] = 1.0
            assert torch.allclose(sv, expected, atol=1e-3)
