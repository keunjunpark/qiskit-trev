"""Full contraction measurement: contract tensor ring to statevector.

Ported from TREV measure/contraction.py (real_form_autograd branch).
"""

import numpy as np
import torch
from torch import Tensor

from ..hamiltonian import Hamiltonian


def contract_tensor_ring(psi: Tensor) -> Tensor:
    """Contract a tensor ring to a full amplitude tensor.

    Args:
        psi: (N, chi, chi, 2) tensor ring.

    Returns:
        Amplitude tensor of shape (2,) * N.
    """
    N = psi.shape[0]

    if N == 1:
        return torch.einsum('iid->d', psi[0])

    psi_new = psi[0]

    for i in range(1, N - 1):
        psi_new = torch.tensordot(psi_new, psi[i], dims=([1], [0]))
        psi_new = torch.movedim(psi_new, -2, 1)

    psi_new = torch.tensordot(psi_new, psi[-1], dims=([0, 1], [1, 0]))
    return psi_new


def measure(tensor: Tensor) -> np.ndarray:
    """Compute probability distribution over all computational basis states.

    Args:
        tensor: (N, chi, chi, 2) tensor ring.

    Returns:
        (2**N,) numpy array of probabilities.
    """
    psi = contract_tensor_ring(tensor)
    prob = (psi * psi.conj()).real
    return prob.reshape(-1).detach().cpu().numpy()


def batched_contract_tensor_ring(batch_psi: Tensor) -> Tensor:
    """Contract a batch of tensor rings to full amplitude tensors.

    Args:
        batch_psi: (B, N, chi, chi, 2) batched tensor ring states.

    Returns:
        Amplitude tensor of shape (B, 2, 2, ..., 2) with N physical dims.
    """
    B, N = batch_psi.shape[:2]

    if N == 1:
        return torch.einsum('biid->bd', batch_psi[:, 0])

    # Start with first site: (B, chi_L0, chi_R0, d0)
    acc = batch_psi[:, 0]

    # Contract intermediate sites
    for i in range(1, N - 1):
        site = batch_psi[:, i]  # (B, chi_L, chi_R, 2)
        chi0 = acc.shape[1]
        chi_prev = acc.shape[2]
        acc_flat = acc.reshape(B, chi0, chi_prev, -1)
        # Contract chi_R_prev with chi_L_i: (B, chi0, chi_R_i, D*2)
        result = torch.einsum('bicD,bcrk->birDk', acc_flat, site)
        acc = result.reshape(B, chi0, site.shape[2], -1)

    # Final site: close the ring (chi_R_{N-1} = chi_L0, chi_L_{N-1} = chi_R_{N-2})
    site = batch_psi[:, -1]  # (B, chi_L, chi_R, 2)
    acc_flat = acc.reshape(B, acc.shape[1], acc.shape[2], -1)
    result = torch.einsum('blrD,brlk->bDk', acc_flat, site)

    return result.reshape(B, *([2] * N))


def batched_expectation_value(batch_tensor: Tensor, hamiltonian: Hamiltonian) -> Tensor:
    """Compute <psi|H|psi> for a batch of tensor ring states via full contraction.

    Builds full statevectors then computes batched quadratic form with H matrix.

    Args:
        batch_tensor: (B, N, chi, chi, 2) batched tensor ring states.
        hamiltonian: Hamiltonian operator.

    Returns:
        (B,) tensor of real expectation values.
    """
    B = batch_tensor.shape[0]
    amps = batched_contract_tensor_ring(batch_tensor).reshape(B, -1)
    H_mat = hamiltonian.get_density_matrix().to(
        device=batch_tensor.device, dtype=batch_tensor.dtype
    )
    ev = torch.einsum('bi,ij,bj->b', amps.conj(), H_mat, amps)
    return ev.real


def expectation_value(tensor: Tensor, hamiltonian: Hamiltonian) -> float:
    """Compute <psi|H|psi> via full contraction.

    Builds the full statevector and Hamiltonian matrix.

    Args:
        tensor: (N, chi, chi, 2) tensor ring.
        hamiltonian: Hamiltonian operator.

    Returns:
        Real expectation value.
    """
    amps = contract_tensor_ring(tensor).reshape(-1)
    H_mat = hamiltonian.get_density_matrix().to(device=tensor.device, dtype=tensor.dtype)
    ev = (amps.conj() @ H_mat @ amps).real
    return ev.item()
