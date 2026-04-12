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


def apply_single_qubit_gate_batch(gate_matrix_batch: Tensor, core_batch: Tensor) -> Tensor:
    """Apply single-qubit gates to a batch of tensor ring cores.

    Args:
        gate_matrix_batch: (B, 2, 2) batch of gate matrices.
        core_batch: (B, chi1, chi2, 2) batch of cores.

    Returns:
        Updated batch of cores, shape (B, chi1, chi2, 2).
    """
    # (B, 2, [2]) @ (B, chi1, chi2, [2]) → (B, 2, chi1, chi2)
    result = torch.einsum('bij,bklj->bikl', gate_matrix_batch, core_batch)
    # → (B, chi1, chi2, 2)
    return result.permute(0, 2, 3, 1)


def apply_double_qubit_gate_batch(
    gate_matrix: Tensor,
    core_a_batch: Tensor,
    core_b_batch: Tensor,
    max_rank: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply a two-qubit gate to a batch of adjacent tensor ring core pairs via SVD.

    Args:
        gate_matrix: (4, 4) or (B, 4, 4) gate matrix.
        core_a_batch: (B, chi1, chi2, 2) batch of cores for qubit a.
        core_b_batch: (B, chi2, chi3, 2) batch of cores for qubit b.
        max_rank: Max bond dimension. Defaults to chi1.

    Returns:
        Tuple of updated batched cores.
    """
    B, chi1, chi2, _ = core_a_batch.shape
    _, chi2_, chi3, _ = core_b_batch.shape
    assert chi2 == chi2_, "Bond mismatch between the two site tensors"

    if max_rank is None:
        max_rank = min(chi1, chi3)

    # Merge: (B, chi1, [chi2], 2) × (B, [chi2], chi3, 2) → (B, chi1, 2, chi3, 2)
    mps = torch.einsum('bikp,bkjq->bijpq', core_a_batch, core_b_batch)

    # Apply gate
    g = gate_matrix
    if g.ndim == 2:
        g = g.expand(B, -1, -1)
    else:
        assert g.shape[0] == B

    mps = mps.reshape(B, chi1 * chi3, 4)
    mps = torch.bmm(mps, g.transpose(1, 2))
    mps = mps.view(B, chi1, chi3, 2, 2)

    mps = mps.permute(0, 3, 1, 4, 2).reshape(B, 2 * chi1, 2 * chi3)

    # Batched SVD
    u, s, vh = torch.linalg.svd(mps)
    k = min(max_rank, s.shape[-1])
    x = u[:, :, :k]
    sx = torch.diag_embed(s[:, :k]).to(torch.cfloat)
    y = vh[:, :k, :]

    new_a = torch.bmm(x, sx).reshape(B, 2, chi1, k).permute(0, 2, 3, 1)
    new_b = y.reshape(B, k, 2, chi3).permute(0, 1, 3, 2)

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
