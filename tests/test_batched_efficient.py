"""Tests for batched efficient contraction expectation values."""

import pytest
import torch

from qiskit_trev.hamiltonian import Hamiltonian
from qiskit_trev.measure.efficient_contraction import (
    expectation_value,
    batched_expectation_value,
)


def _random_tensor_ring(B, N, chi, dtype=torch.cfloat, seed=42):
    """Create a random batch of tensor ring states."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, chi, chi, 2, dtype=dtype, generator=gen)


class TestBatchedEfficientCorrectness:
    """Batched results must match sequential element-by-element results."""

    @pytest.mark.parametrize("B,N,chi", [(1, 4, 4), (5, 4, 8), (10, 6, 4)])
    def test_single_z_term(self, B, N, chi):
        """Single Z on first qubit: batched[i] == sequential[i]."""
        pauli = "Z" + "I" * (N - 1)
        ham = Hamiltonian.from_pauli_list([(pauli, 1.0)])
        batch = _random_tensor_ring(B, N, chi)

        batched = batched_expectation_value(batch, ham)
        sequential = torch.tensor(
            [expectation_value(batch[i], ham) for i in range(B)]
        )

        assert batched.shape == (B,)
        torch.testing.assert_close(batched, sequential, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("B,N,chi", [(3, 4, 4), (7, 4, 8)])
    def test_multi_zi_terms(self, B, N, chi):
        """Multiple Z/I terms: ZZII + IIZZ + ZIZI."""
        ham = Hamiltonian.from_pauli_list([
            ("Z" * 2 + "I" * (N - 2), 1.0),
            ("I" * 2 + "Z" * 2 + "I" * (N - 4) if N > 4 else "I" * 2 + "Z" * (N - 2), 0.5),
            ("ZI" * (N // 2) + "I" * (N % 2), -0.3),
        ])
        batch = _random_tensor_ring(B, N, chi)

        batched = batched_expectation_value(batch, ham)
        sequential = torch.tensor(
            [expectation_value(batch[i], ham) for i in range(B)]
        )

        torch.testing.assert_close(batched, sequential, atol=1e-5, rtol=1e-5)

    def test_batch_of_one(self):
        """B=1 edge case."""
        N, chi = 4, 4
        ham = Hamiltonian.from_pauli_list([("ZIZI", 1.0)])
        batch = _random_tensor_ring(1, N, chi)

        batched = batched_expectation_value(batch, ham)
        single = expectation_value(batch[0], ham)

        assert batched.shape == (1,)
        assert abs(batched[0].item() - single) < 1e-5

    def test_single_qubit(self):
        """N=1 edge case."""
        B, chi = 5, 4
        ham = Hamiltonian.from_pauli_list([("Z", 1.0)])
        batch = _random_tensor_ring(B, 1, chi)

        batched = batched_expectation_value(batch, ham)
        sequential = torch.tensor(
            [expectation_value(batch[i], ham) for i in range(B)]
        )

        torch.testing.assert_close(batched, sequential, atol=1e-5, rtol=1e-5)

    def test_all_identity(self):
        """All-I Hamiltonian: expectation = trace(rho) = norm^2."""
        N, chi, B = 4, 4, 3
        ham = Hamiltonian.from_pauli_list([("IIII", 1.0)])
        batch = _random_tensor_ring(B, N, chi)

        batched = batched_expectation_value(batch, ham)
        sequential = torch.tensor(
            [expectation_value(batch[i], ham) for i in range(B)]
        )

        torch.testing.assert_close(batched, sequential, atol=1e-5, rtol=1e-5)


class TestBatchedEfficientChunking:
    """Test chunk_size parameter for Hamiltonian term chunking."""

    def test_chunked_matches_unchunked(self):
        """Chunking Hamiltonian terms should give same result."""
        N, chi, B = 4, 4, 5
        ham = Hamiltonian.from_pauli_list([
            ("ZZII", 1.0),
            ("IIZZ", 0.5),
            ("ZIZI", -0.3),
            ("IZIZ", 0.7),
        ])
        batch = _random_tensor_ring(B, N, chi)

        full = batched_expectation_value(batch, ham)
        chunked = batched_expectation_value(batch, ham, chunk_size=2)

        torch.testing.assert_close(full, chunked, atol=1e-5, rtol=1e-5)


class TestBatchedEfficientOutput:
    """Test output properties."""

    def test_output_is_real(self):
        """Output should be real-valued float tensor."""
        N, chi, B = 4, 4, 3
        ham = Hamiltonian.from_pauli_list([("ZIII", 1.0)])
        batch = _random_tensor_ring(B, N, chi)

        result = batched_expectation_value(batch, ham)
        assert result.is_floating_point()
        assert not result.is_complex()
