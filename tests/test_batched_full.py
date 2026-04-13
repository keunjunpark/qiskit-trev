"""Tests for batched full contraction expectation values."""

import pytest
import torch

from qiskit_trev.hamiltonian import Hamiltonian
from qiskit_trev.measure.full_contraction import (
    contract_tensor_ring,
    expectation_value,
    batched_contract_tensor_ring,
    batched_expectation_value,
)


def _random_tensor_ring(B, N, chi, dtype=torch.cfloat, seed=42):
    """Create a random batch of tensor ring states."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, chi, chi, 2, dtype=dtype, generator=gen)


class TestBatchedContractTensorRing:
    """batched_contract_tensor_ring must match element-wise contract_tensor_ring."""

    @pytest.mark.parametrize("B,N,chi", [(1, 2, 4), (5, 3, 4), (3, 4, 8)])
    def test_matches_sequential(self, B, N, chi):
        batch = _random_tensor_ring(B, N, chi)

        batched = batched_contract_tensor_ring(batch)  # (B, 2, 2, ..., 2)
        for i in range(B):
            single = contract_tensor_ring(batch[i])  # (2, 2, ..., 2)
            torch.testing.assert_close(batched[i], single, atol=1e-5, rtol=1e-5)

    def test_single_qubit(self):
        """N=1 edge case."""
        batch = _random_tensor_ring(3, 1, 4)
        batched = batched_contract_tensor_ring(batch)
        assert batched.shape == (3, 2)
        for i in range(3):
            single = contract_tensor_ring(batch[i])
            torch.testing.assert_close(batched[i], single, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        """Output should be (B, 2, 2, ..., 2) with N dimensions of size 2."""
        B, N, chi = 4, 3, 4
        batch = _random_tensor_ring(B, N, chi)
        result = batched_contract_tensor_ring(batch)
        assert result.shape == (B,) + (2,) * N


class TestBatchedFullExpectationValue:
    """batched_expectation_value must match sequential expectation_value."""

    @pytest.mark.parametrize("B,N,chi", [(1, 3, 4), (5, 3, 4), (3, 4, 4)])
    def test_zi_hamiltonian(self, B, N, chi):
        """Z/I only Hamiltonian via full contraction."""
        pauli = "Z" + "I" * (N - 1)
        ham = Hamiltonian.from_pauli_list([(pauli, 1.0)])
        batch = _random_tensor_ring(B, N, chi)

        batched = batched_expectation_value(batch, ham)
        sequential = torch.tensor(
            [expectation_value(batch[i], ham) for i in range(B)]
        )

        assert batched.shape == (B,)
        torch.testing.assert_close(batched, sequential, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("B,N,chi", [(3, 3, 4), (5, 4, 4)])
    def test_xy_hamiltonian(self, B, N, chi):
        """Hamiltonian with X and Y terms."""
        ham = Hamiltonian.from_pauli_list([
            ("X" + "I" * (N - 1), 1.0),
            ("I" + "Y" + "I" * (N - 2), 0.5),
            ("Z" + "I" * (N - 1), -0.3),
        ])
        batch = _random_tensor_ring(B, N, chi)

        batched = batched_expectation_value(batch, ham)
        sequential = torch.tensor(
            [expectation_value(batch[i], ham) for i in range(B)]
        )

        torch.testing.assert_close(batched, sequential, atol=1e-4, rtol=1e-4)

    def test_output_is_real(self):
        """Output should be real-valued float tensor."""
        ham = Hamiltonian.from_pauli_list([("XYZ", 1.0)])
        batch = _random_tensor_ring(3, 3, 4)

        result = batched_expectation_value(batch, ham)
        assert result.is_floating_point()
        assert not result.is_complex()
