"""Shared test utilities for qiskit-trev tests."""

import torch
import numpy as np


def contract_tensor_ring(psi: torch.Tensor) -> torch.Tensor:
    """Contract a tensor ring to a full amplitude tensor.

    Ported from TREV measure/contraction.py.

    Args:
        psi: Tensor ring of shape (N, chi, chi, 2).

    Returns:
        Amplitude tensor of shape (2,) * N.
    """
    N = psi.shape[0]

    if N == 1:
        return torch.einsum('iid->d', psi[0])

    psi_new = psi[0]  # (chi1, chi2, 2)

    for i in range(1, N - 1):
        psi_new = torch.tensordot(psi_new, psi[i], dims=([1], [0]))
        psi_new = torch.movedim(psi_new, -2, 1)

    # Close the ring
    psi_new = torch.tensordot(psi_new, psi[-1], dims=([0, 1], [1, 0]))
    return psi_new


def statevector(tensor: torch.Tensor) -> torch.Tensor:
    """Contract tensor ring and return flattened statevector.

    Args:
        tensor: Tensor ring of shape (N, chi, chi, 2).

    Returns:
        Complex amplitude vector of shape (2**N,).
    """
    amps = contract_tensor_ring(tensor)
    return amps.reshape(-1)


def probabilities(tensor: torch.Tensor) -> np.ndarray:
    """Contract tensor ring and return probability vector.

    Args:
        tensor: Tensor ring of shape (N, chi, chi, 2).

    Returns:
        Real probability array of shape (2**N,).
    """
    sv = statevector(tensor)
    prob = (sv * sv.conj()).real
    return prob.detach().cpu().numpy()
