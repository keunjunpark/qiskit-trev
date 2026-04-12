"""Gate application via SVD contraction for tensor rings.

Ported from TREV gates/contraction.py (real_form_autograd branch).
"""

import warnings

import torch
from torch import Tensor

warnings.filterwarnings("ignore", message="torch.linalg.svd")


def apply_single_qubit_gate(gate_matrix: Tensor, core: Tensor) -> Tensor:
    """Apply a single-qubit gate to a tensor ring core.

    Args:
        gate_matrix: (2, 2) unitary gate matrix.
        core: (chi1, chi2, 2) tensor ring core.

    Returns:
        Updated core of shape (chi1, chi2, 2).
    """
    # (2, [2]) . (chi1, chi2, [2]) = (2, chi1, chi2)
    result = torch.tensordot(gate_matrix, core, ([1], [2]))
    # (chi1, chi2, 2)
    return torch.moveaxis(result, 0, 2)


def apply_double_qubit_gate(
    gate_matrix: Tensor,
    core_a: Tensor,
    core_b: Tensor,
    max_rank: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply a two-qubit gate to adjacent tensor ring cores via SVD.

    Args:
        gate_matrix: (4, 4) unitary gate matrix.
        core_a: (chi1, chi2, 2) tensor ring core for qubit a.
        core_b: (chi2, chi3, 2) tensor ring core for qubit b.
        max_rank: Maximum bond dimension after SVD truncation.
            If None, defaults to min(chi1, chi3) (input bond dims).

    Returns:
        Tuple of updated cores (core_a', core_b').
    """
    chi1 = core_a.shape[0]
    chi3 = core_b.shape[1]

    if max_rank is None:
        max_rank = min(chi1, chi3)

    # Merge: contract shared bond (axis 1 of a with axis 0 of b)
    # (chi1, [chi2], 2) . ([chi2], chi3, 2) = (chi1, 2, chi3, 2)
    mps = torch.tensordot(core_a, core_b, ([1], [0]))
    # Reorder to (chi1, chi3, 2, 2)
    mps = torch.moveaxis(mps, 2, 1)

    # Apply gate
    gate_tensor = gate_matrix.reshape(2, 2, 2, 2)
    # (2, 2, [2], [2]) . (chi1, chi3, [2], [2]) = (2, 2, chi1, chi3)
    mps = torch.tensordot(gate_tensor, mps, ([2, 3], [2, 3]))
    # Reshape to (2*chi1, 2*chi3) for SVD
    # First reorder: (2, chi1, 2, chi3) then reshape
    mps = torch.moveaxis(mps, 1, 2).reshape(chi1 * 2, chi3 * 2)

    # SVD and truncate
    u, s, v = torch.linalg.svd(mps, full_matrices=False)
    k = min(max_rank, len(s))
    x = u[:, :k]
    sx = torch.diag(s[:k]).to(torch.cfloat)
    y = v[:k, :]

    # Split back into two cores
    new_a = torch.mm(x, sx).reshape(2, chi1, k)
    new_b = y.reshape(k, 2, chi3)

    new_a = torch.moveaxis(new_a, 0, 2)  # (chi1, k, 2)
    new_b = torch.moveaxis(new_b, 1, 2)  # (k, chi3, 2)

    return new_a, new_b


def swap_gate_matrix(matrix: Tensor) -> Tensor:
    """Permute qubit labels in a 4x4 gate: SWAP @ matrix @ SWAP.

    Args:
        matrix: (4, 4) gate matrix.

    Returns:
        (4, 4) matrix with swapped qubit labels.
    """
    swap = torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=matrix.dtype,
        device=matrix.device,
    )
    return swap @ matrix @ swap
