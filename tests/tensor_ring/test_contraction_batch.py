"""Tests for batched gate contraction functions."""

import math
import pytest
import torch

from qiskit_trev.tensor_ring.gates import H, X, RX, RY, CNOT, SWAP, I
from qiskit_trev.tensor_ring.contraction import (
    apply_single_qubit_gate,
    apply_double_qubit_gate,
    apply_single_qubit_gate_batch,
    apply_double_qubit_gate_batch,
)
from tests.utils import statevector


def _make_zero_state(num_qubits, rank):
    tensor = torch.zeros((num_qubits, rank, rank, 2), dtype=torch.cfloat)
    tensor[:, 0, 0, 0] = 1.0
    return tensor


def _make_batch_zero_state(batch_size, num_qubits, rank):
    tensor = torch.zeros((num_qubits, rank, rank, 2), dtype=torch.cfloat)
    tensor[:, 0, 0, 0] = 1.0
    return tensor.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).clone()


class TestApplySingleQubitGateBatch:

    def test_output_shape(self):
        """Batch of cores maintains shape."""
        cores = torch.randn(3, 4, 4, 2, dtype=torch.cfloat)  # (B, chi, chi, 2)
        gate = H()  # (2, 2)
        result = apply_single_qubit_gate_batch(gate.unsqueeze(0).expand(3, -1, -1), cores)
        assert result.shape == (3, 4, 4, 2)

    def test_matches_sequential(self):
        """Batch result should match applying gate to each core individually."""
        B = 4
        tensor = _make_batch_zero_state(B, 2, 4)
        gate = H()

        # Sequential
        results_seq = []
        for b in range(B):
            results_seq.append(apply_single_qubit_gate(gate, tensor[b, 0]))

        # Batch
        gate_batch = gate.unsqueeze(0).expand(B, -1, -1)
        result_batch = apply_single_qubit_gate_batch(gate_batch, tensor[:, 0])

        for b in range(B):
            assert torch.allclose(result_batch[b], results_seq[b], atol=1e-6)

    def test_different_gates_per_batch(self):
        """Each batch element can have a different gate."""
        B = 3
        cores = _make_batch_zero_state(B, 1, 1)[:, 0]  # (B, 1, 1, 2)
        gates = torch.stack([RX(0.0), RX(math.pi / 2), RX(math.pi)])  # (B, 2, 2)
        result = apply_single_qubit_gate_batch(gates, cores)
        assert result.shape == (3, 1, 1, 2)
        # At theta=0: |0> → |0>
        assert torch.abs(result[0, 0, 0, 0]).item() > 0.99
        # At theta=pi: |0> → -i|1>
        assert torch.abs(result[2, 0, 0, 1]).item() > 0.99


class TestApplyDoubleQubitGateBatch:

    def test_output_shape(self):
        B, chi = 3, 4
        core_a = torch.randn(B, chi, chi, 2, dtype=torch.cfloat)
        core_b = torch.randn(B, chi, chi, 2, dtype=torch.cfloat)
        new_a, new_b = apply_double_qubit_gate_batch(CNOT(), core_a, core_b)
        assert new_a.shape[0] == B
        assert new_a.shape[3] == 2
        assert new_b.shape[3] == 2

    def test_matches_sequential_cnot(self):
        """Batch CNOT should match sequential application."""
        B = 4
        tensor = _make_batch_zero_state(B, 2, 4)

        # Apply X to qubit 0 for batch elements 2,3 (to get |10>)
        for b in [2, 3]:
            tensor[b, 0] = apply_single_qubit_gate(X(), tensor[b, 0])

        # Sequential CNOT
        seq_a, seq_b = [], []
        for b in range(B):
            a, b_out = apply_double_qubit_gate(CNOT(), tensor[b, 0], tensor[b, 1])
            seq_a.append(a)
            seq_b.append(b_out)

        # Batch CNOT
        batch_a, batch_b = apply_double_qubit_gate_batch(
            CNOT(), tensor[:, 0], tensor[:, 1]
        )

        for b in range(B):
            assert torch.allclose(batch_a[b], seq_a[b], atol=1e-5), f"core_a mismatch at batch {b}"
            assert torch.allclose(batch_b[b], seq_b[b], atol=1e-5), f"core_b mismatch at batch {b}"

    def test_bell_state_batch(self):
        """Create bell states in batch."""
        B = 2
        tensor = _make_batch_zero_state(B, 2, 4)

        # Apply H to qubit 0
        h_batch = H().unsqueeze(0).expand(B, -1, -1)
        tensor[:, 0] = apply_single_qubit_gate_batch(h_batch, tensor[:, 0])

        # Apply CNOT(0,1)
        tensor[:, 0], tensor[:, 1] = apply_double_qubit_gate_batch(
            CNOT(), tensor[:, 0], tensor[:, 1]
        )

        # Each should be a bell state
        for b in range(B):
            sv = statevector(tensor[b])
            expected = torch.tensor([1, 0, 0, 1], dtype=torch.cfloat) / math.sqrt(2)
            assert torch.allclose(sv, expected, atol=1e-4)

    def test_batched_gate_matrix(self):
        """Gate matrix can be (B, 4, 4) for per-batch gates."""
        B = 2
        tensor = _make_batch_zero_state(B, 2, 4)
        gates = CNOT().unsqueeze(0).expand(B, -1, -1)
        new_a, new_b = apply_double_qubit_gate_batch(gates, tensor[:, 0], tensor[:, 1])
        assert new_a.shape[0] == B
