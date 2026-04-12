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
