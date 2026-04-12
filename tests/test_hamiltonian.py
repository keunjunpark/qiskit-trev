"""Tests for Hamiltonian class."""

import pytest
import torch
import numpy as np

from qiskit_trev.hamiltonian import Hamiltonian


class TestConstruction:

    def test_from_pauli_list(self):
        h = Hamiltonian.from_pauli_list([("ZZ", 1.0), ("IX", 0.5)])
        assert h.num_qubits == 2
        assert h.paulis == ["ZZ", "IX"]
        assert h.coefficients == [1.0, 0.5]

    def test_from_pauli_list_empty(self):
        h = Hamiltonian.from_pauli_list([])
        assert h.num_qubits == 0
        assert len(h.paulis) == 0

    def test_add_pauli(self):
        h = Hamiltonian(num_qubits=2)
        h.add_pauli("ZI", 1.0)
        h.add_pauli("IZ", -0.5)
        assert len(h.paulis) == 2
        assert h.coefficients == [1.0, -0.5]

    def test_add_pauli_wrong_length(self):
        h = Hamiltonian(num_qubits=2)
        with pytest.raises(ValueError, match="length"):
            h.add_pauli("ZZZ", 1.0)

    def test_add_pauli_invalid_char(self):
        h = Hamiltonian(num_qubits=2)
        with pytest.raises(ValueError, match="Invalid"):
            h.add_pauli("ZA", 1.0)


class TestBoolPauliTensor:

    def test_single_term(self):
        h = Hamiltonian.from_pauli_list([("ZZII", 1.0)])
        t = h.get_bool_pauli_tensor()
        assert t.shape == (1, 4)
        assert t[0].tolist() == [True, True, False, False]

    def test_multi_term(self):
        h = Hamiltonian.from_pauli_list([("ZI", 1.0), ("IZ", 0.5)])
        t = h.get_bool_pauli_tensor()
        assert t.shape == (2, 2)
        assert t[0].tolist() == [True, False]
        assert t[1].tolist() == [False, True]

    def test_identity_all_false(self):
        h = Hamiltonian.from_pauli_list([("II", 1.0)])
        t = h.get_bool_pauli_tensor()
        assert not t.any()


class TestPauliOpTensor:

    def test_encoding(self):
        """I=0, X=1, Y=2, Z=3."""
        h = Hamiltonian.from_pauli_list([("IXYZ", 1.0)])
        t = h.get_pauli_op_tensor()
        assert t.shape == (1, 4)
        assert t[0].tolist() == [0, 1, 2, 3]

    def test_multi_term(self):
        h = Hamiltonian.from_pauli_list([("ZZ", 1.0), ("XI", 0.5)])
        t = h.get_pauli_op_tensor()
        assert t[0].tolist() == [3, 3]
        assert t[1].tolist() == [1, 0]


class TestQWCGroups:

    def test_single_term(self):
        h = Hamiltonian.from_pauli_list([("ZZ", 1.0)])
        groups = h.get_qwc_groups()
        assert len(groups) == 1
        assert groups[0]['term_indices'] == [0]
        assert groups[0]['basis'] == "ZZ"

    def test_compatible_terms_grouped(self):
        """ZI and IZ are QWC-compatible (both measure Z)."""
        h = Hamiltonian.from_pauli_list([("ZI", 1.0), ("IZ", 0.5)])
        groups = h.get_qwc_groups()
        assert len(groups) == 1
        assert set(groups[0]['term_indices']) == {0, 1}

    def test_incompatible_terms_separate(self):
        """ZI and XI need different bases."""
        h = Hamiltonian.from_pauli_list([("ZI", 1.0), ("XI", 0.5)])
        groups = h.get_qwc_groups()
        assert len(groups) == 2

    def test_complex_grouping(self):
        """ZZII, IZZI, IIZZ should all be in same group."""
        h = Hamiltonian.from_pauli_list([
            ("ZZII", 1.0), ("IZZI", 0.5), ("IIZZ", 0.5),
        ])
        groups = h.get_qwc_groups()
        assert len(groups) == 1
        assert groups[0]['basis'] == "ZZZZ"


class TestDensityMatrix:

    def test_single_Z(self):
        h = Hamiltonian.from_pauli_list([("Z", 1.0)])
        m = h.get_density_matrix()
        expected = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)
        assert torch.allclose(m, expected)

    def test_ZZ(self):
        h = Hamiltonian.from_pauli_list([("ZZ", 1.0)])
        m = h.get_density_matrix()
        expected = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.cfloat))
        assert torch.allclose(m, expected)

    def test_multi_term(self):
        h = Hamiltonian.from_pauli_list([("ZI", 1.0), ("IZ", 0.5)])
        m = h.get_density_matrix()
        # ZI = Z⊗I = diag(1,1,-1,-1), IZ = I⊗Z = diag(1,-1,1,-1)
        expected = torch.diag(torch.tensor([1.5, 0.5, -0.5, -1.5], dtype=torch.cfloat))
        assert torch.allclose(m, expected)

    def test_hermitian(self):
        h = Hamiltonian.from_pauli_list([("XY", 0.5), ("YX", 0.5)])
        m = h.get_density_matrix()
        assert torch.allclose(m, m.conj().T, atol=1e-6)


class TestHasOnlyZI:

    def test_only_zi(self):
        h = Hamiltonian.from_pauli_list([("ZI", 1.0), ("IZ", 0.5)])
        assert h.has_only_zi

    def test_with_x(self):
        h = Hamiltonian.from_pauli_list([("XI", 1.0)])
        assert not h.has_only_zi

    def test_identity_only(self):
        h = Hamiltonian.from_pauli_list([("II", 1.0)])
        assert h.has_only_zi


class TestPermuted:

    def test_simple_permutation(self):
        h = Hamiltonian.from_pauli_list([("ZI", 1.0)])
        p = h.permuted([1, 0])  # swap qubits
        assert p.paulis == ["IZ"]
        assert p.coefficients == [1.0]

    def test_3qubit_permutation(self):
        h = Hamiltonian.from_pauli_list([("ZIX", 1.0)])
        p = h.permuted([2, 0, 1])
        assert p.paulis == ["IXZ"]


class TestRotateTensor:

    def test_z_basis_no_rotation(self):
        """Z/I basis should not change the tensor."""
        tensor = torch.randn(2, 4, 4, 2, dtype=torch.cfloat)
        from qiskit_trev.hamiltonian import rotate_tensor_for_measurement
        rotated = rotate_tensor_for_measurement(tensor, "ZI")
        assert torch.allclose(rotated, tensor)

    def test_x_basis_applies_hadamard(self):
        """X basis should apply Hadamard rotation."""
        tensor = torch.zeros(1, 1, 1, 2, dtype=torch.cfloat)
        tensor[0, 0, 0, 0] = 1.0  # |0>
        from qiskit_trev.hamiltonian import rotate_tensor_for_measurement
        rotated = rotate_tensor_for_measurement(tensor, "X")
        # H|0> = |+>, so core should have equal amplitudes
        assert abs(rotated[0, 0, 0, 0].item()) > 0.1
        assert abs(rotated[0, 0, 0, 1].item()) > 0.1

    def test_rotation_does_not_modify_original(self):
        tensor = torch.randn(2, 4, 4, 2, dtype=torch.cfloat)
        original = tensor.clone()
        from qiskit_trev.hamiltonian import rotate_tensor_for_measurement
        _ = rotate_tensor_for_measurement(tensor, "XY")
        assert torch.allclose(tensor, original)

    def test_batched_rotation(self):
        """5D input (B, N, chi, chi, 2) should work."""
        tensor = torch.randn(3, 2, 4, 4, 2, dtype=torch.cfloat)
        from qiskit_trev.hamiltonian import rotate_tensor_for_measurement
        rotated = rotate_tensor_for_measurement(tensor, "XI")
        assert rotated.shape == tensor.shape
