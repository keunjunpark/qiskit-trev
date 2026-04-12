"""Efficient contraction measurement: exact expectation via double-layer transfer matrices.

Ported from TREV measure/efficient_contraction.py (real_form_autograd branch).

Computes <psi|H|psi> exactly using batched double-layer transfer matrix
contraction. Only supports Z/I Hamiltonians. Works for both chain and
ring (periodic) topology.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..hamiltonian import Hamiltonian


@torch.no_grad()
def expectation_value(
    tensor: Tensor,
    hamiltonian: Hamiltonian,
    chunk_size: int | None = None,
) -> float:
    """Compute <psi|H|psi> via batched double-layer transfer matrix contraction.

    Batches Hamiltonian terms and contracts through all sites using
    double-layer transfer matrices E_I and E_Z per site.

    Args:
        tensor: (N, chi, chi, 2) tensor ring.
        hamiltonian: Hamiltonian (Z/I terms only).
        chunk_size: Process terms in chunks (None = all at once).

    Returns:
        Real expectation value.
    """
    device = tensor.device
    N = tensor.shape[0]

    paulis = hamiltonian.get_bool_pauli_tensor().to(device)  # (T, N)
    T = paulis.shape[0]

    coeffs = torch.as_tensor(
        hamiltonian.coefficients, dtype=torch.cfloat, device=device
    )

    Z_op = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=device)
    I_op = torch.eye(2, dtype=torch.cfloat, device=device)

    if chunk_size is None:
        chunk_size = T

    total = torch.zeros((), dtype=torch.cfloat, device=device)

    for start in range(0, T, chunk_size):
        stop = min(start + chunk_size, T)
        mask = paulis[start:stop]      # (B, N)
        coefs = coeffs[start:stop]     # (B,)
        B = mask.size(0)

        # Build first site's batched transfer matrix
        A = tensor[0]  # (chi_L, chi_R, 2)
        AO_I = torch.einsum('lrd,dk->lrk', A, I_op)
        AO_Z = torch.einsum('lrd,dk->lrk', A, Z_op)
        E_I = torch.tensordot(A.conj(), AO_I, dims=([2], [2])).permute(0, 2, 1, 3)
        E_Z = torch.tensordot(A.conj(), AO_Z, dims=([2], [2])).permute(0, 2, 1, 3)

        m0 = mask[:, 0].view(B, 1, 1, 1, 1)
        ten = torch.where(m0, E_Z.unsqueeze(0), E_I.unsqueeze(0))  # (B, l, l', r, r')

        # Contract remaining sites
        for i in range(1, N):
            A = tensor[i]
            AO_I = torch.einsum('lrd,dk->lrk', A, I_op)
            AO_Z = torch.einsum('lrd,dk->lrk', A, Z_op)
            Ei_I = torch.tensordot(A.conj(), AO_I, dims=([2], [2])).permute(0, 2, 1, 3)
            Ei_Z = torch.tensordot(A.conj(), AO_Z, dims=([2], [2])).permute(0, 2, 1, 3)

            mi = mask[:, i].view(B, 1, 1, 1, 1)
            Ei = torch.where(mi, Ei_Z.unsqueeze(0), Ei_I.unsqueeze(0))  # (B, l, l', r, r')

            # Contract: ten(b,i,j,p,q) @ Ei(b,p,q,r,s) → (b,i,j,r,s)
            ten = torch.einsum('bijpq,bpqrs->bijrs', ten, Ei)

        # Close ring: trace over (i=r, j=s)
        vals = torch.einsum('bijij->b', ten)  # (B,)
        total = total + torch.sum(coefs * vals)

    return total.real.item()
