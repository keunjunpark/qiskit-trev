"""Hamiltonian representation and measurement basis rotation utilities.

Ported from TREV hamiltonian/hamiltonian.py (real_form_autograd branch).
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch import Tensor

from .tensor_ring.gates import I, X, Y, Z

_OPS = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
_VALID_OPS = frozenset('IXYZ')
_OP_TO_UINT8 = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}


class Hamiltonian:
    """Pauli string Hamiltonian: H = sum_i c_i * P_i.

    Each term is a Pauli string (e.g., "ZZII") with a coefficient.
    """

    def __init__(
        self,
        num_qubits: int = 0,
        paulis: List[str] | None = None,
        coefficients: List[complex] | None = None,
    ):
        self.num_qubits = num_qubits
        self.paulis: List[str] = paulis if paulis is not None else []
        self.coefficients: List[complex] = coefficients if coefficients is not None else []

    @classmethod
    def from_pauli_list(cls, terms: list[tuple[str, complex]]) -> Hamiltonian:
        """Create Hamiltonian from list of (pauli_string, coefficient) tuples."""
        if not terms:
            return cls(num_qubits=0)
        paulis = [t[0] for t in terms]
        coefficients = [t[1] for t in terms]
        num_qubits = len(paulis[0])
        return cls(num_qubits=num_qubits, paulis=paulis, coefficients=coefficients)

    def add_pauli(self, pauli: str, coefficient: complex) -> None:
        if len(pauli) != self.num_qubits:
            raise ValueError(
                f"Pauli string length {len(pauli)} != num_qubits {self.num_qubits}"
            )
        for ch in pauli:
            if ch not in _VALID_OPS:
                raise ValueError(
                    f"Invalid Pauli operator '{ch}', must be one of {_VALID_OPS}"
                )
        self.paulis.append(pauli)
        self.coefficients.append(coefficient)

    def get_bool_pauli_tensor(self) -> Tensor:
        """Return (T, N) bool tensor. True where Pauli is Z."""
        return torch.tensor(
            [[p[i] == 'Z' for i in range(self.num_qubits)] for p in self.paulis],
            dtype=torch.bool,
        )

    def get_pauli_op_tensor(self) -> Tensor:
        """Return (T, N) uint8 tensor. I=0, X=1, Y=2, Z=3."""
        return torch.tensor(
            [[_OP_TO_UINT8[ch] for ch in p] for p in self.paulis],
            dtype=torch.uint8,
        )

    @property
    def has_only_zi(self) -> bool:
        """True if all Pauli strings contain only Z and I."""
        return all(ch in ('Z', 'I') for p in self.paulis for ch in p)

    def permuted(self, perm: list[int]) -> Hamiltonian:
        """Return new Hamiltonian with qubit indices permuted.

        perm[logical] = physical.
        """
        new_paulis = []
        for p in self.paulis:
            chars = ['I'] * self.num_qubits
            for i, ch in enumerate(p):
                chars[perm[i]] = ch
            new_paulis.append(''.join(chars))
        return Hamiltonian(self.num_qubits, new_paulis, list(self.coefficients))

    def get_qwc_groups(self) -> list[dict]:
        """Group terms by qubit-wise commutativity.

        Returns list of dicts with:
            'term_indices': list of int
            'basis': str of length num_qubits
        """
        groups: list[tuple[list[int], list[str]]] = []

        for t, pauli in enumerate(self.paulis):
            placed = False
            for g_indices, g_basis in groups:
                compatible = True
                for q in range(self.num_qubits):
                    if pauli[q] == 'I' or g_basis[q] == 'I':
                        continue
                    if pauli[q] != g_basis[q]:
                        compatible = False
                        break
                if compatible:
                    g_indices.append(t)
                    for q in range(self.num_qubits):
                        if g_basis[q] == 'I' and pauli[q] != 'I':
                            g_basis[q] = pauli[q]
                    placed = True
                    break

            if not placed:
                groups.append(([t], list(pauli)))

        return [{'term_indices': idx, 'basis': ''.join(b)} for idx, b in groups]

    def get_density_matrix(self) -> Tensor:
        """Build the full 2^N x 2^N Hamiltonian matrix."""
        dim = 2 ** self.num_qubits
        rho = torch.zeros((dim, dim), dtype=torch.cfloat)
        for pauli_str, coeff in zip(self.paulis, self.coefficients):
            mat = _OPS[pauli_str[0]]()
            for p in pauli_str[1:]:
                mat = torch.kron(mat, _OPS[p]())
            rho += coeff * mat
        return rho


# --- Measurement basis rotation ---

_S = 2 ** -0.5

_MEAS_ROTATIONS = {
    'X': torch.tensor([[_S, _S], [_S, -_S]], dtype=torch.cfloat),
    'Y': torch.tensor([[_S, -1j * _S], [-1j * _S, _S]], dtype=torch.cfloat),
}


def rotate_tensor_for_measurement(tensor: Tensor, basis: str) -> Tensor:
    """Apply measurement basis rotations to tensor ring cores.

    Args:
        tensor: (N, chi, chi, 2) or (B, N, chi, chi, 2).
        basis: String of length N with chars from 'IXYZ'.

    Returns:
        Rotated tensor (cloned).
    """
    rotated = tensor.clone()
    is_batched = tensor.dim() == 5

    for i, b in enumerate(basis):
        U = _MEAS_ROTATIONS.get(b)
        if U is None:
            continue
        U = U.to(device=tensor.device, dtype=tensor.dtype)
        if is_batched:
            rotated[:, i] = rotated[:, i] @ U.mT
        else:
            rotated[i] = rotated[i] @ U.mT

    return rotated
