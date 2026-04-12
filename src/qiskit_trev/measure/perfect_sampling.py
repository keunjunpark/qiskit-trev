"""Perfect sampling measurement: sequential site-by-site collapse.

Ported from TREV measure/perfect_sampling.py (real_form_autograd branch).

Note: This implementation uses sequential left-to-right sampling which is
correct for chain (open boundary) topology. For ring (periodic) topology,
use right_suffix or efficient_contraction instead.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..hamiltonian import Hamiltonian


def measure(tensor: Tensor, shots: int = 10000, device: str = None) -> list:
    """Sample bitstrings via sequential perfect sampling.

    Args:
        tensor: (N, chi, chi, 2) tensor ring.
        shots: Number of shots.
        device: Torch device.

    Returns:
        List of length 2**N with probability estimates from sampling.
    """
    if device is None:
        device = tensor.device

    q0 = torch.tensor([[1], [0]], dtype=torch.cfloat, device=device)
    q1 = torch.tensor([[0], [1]], dtype=torch.cfloat, device=device)

    N = tensor.size(0)
    prob_dist = [0] * (2 ** N)
    increment = 1 / shots

    for _ in range(int(shots)):
        prev: Optional[Tensor] = None
        key: int = 0

        for i in range(N):
            curr_ten = tensor[i]

            if i == 0:
                qubit_0 = torch.tensordot(curr_ten, q0, ([2], [0])).squeeze(-1)
                qubit_1 = torch.tensordot(curr_ten, q1, ([2], [0])).squeeze(-1)

                prob_0 = torch.tensordot(qubit_0, qubit_0.mH, ([1, 0], [0, 1])).real.item()
                prob_1 = torch.tensordot(qubit_1, qubit_1.mH, ([1, 0], [0, 1])).real.item()
                total = prob_0 + prob_1
            else:
                if prev is not None:
                    curr_ten = torch.tensordot(prev, curr_ten, ([1], [0]))
                qubit_0 = torch.tensordot(curr_ten, q0, ([2], [0])).squeeze(-1)
                qubit_1 = torch.tensordot(curr_ten, q1, ([2], [0])).squeeze(-1)

                prob_0 = torch.tensordot(qubit_0, qubit_0.mH, ([1, 0], [0, 1])).real.item()
                prob_1 = torch.tensordot(qubit_1, qubit_1.mH, ([1, 0], [0, 1])).real.item()
                total = prob_0 + prob_1

            rnd = torch.rand(1).item()
            if total == 0:
                prob_0 = 0.5
                total = 1

            if rnd > prob_0 / total:
                prev = qubit_1
                key += (2 ** i)
            else:
                prev = qubit_0

        prob_dist[key] += increment

    return prob_dist


@torch.no_grad()
def expectation_value(
    tensor: Tensor,
    hamiltonian: Hamiltonian,
    shots: int = 10000,
    device: str = None,
) -> float:
    """Compute <psi|H|psi> via batched perfect sampling with coefficient flipping.

    Only supports Z/I Hamiltonians. Correct for chain topology.

    Args:
        tensor: (N, chi, chi, 2) tensor ring.
        hamiltonian: Hamiltonian (must contain only Z/I terms).
        shots: Number of shots.
        device: Torch device.

    Returns:
        Real expectation value estimate.
    """
    if device is None:
        device = tensor.device

    q0 = torch.tensor([[1], [0]], dtype=torch.cfloat, device=device)
    q1 = torch.tensor([[0], [1]], dtype=torch.cfloat, device=device)

    paulis_tensor = hamiltonian.get_bool_pauli_tensor().to(device=device)
    batch_coefs = (
        torch.tensor(hamiltonian.coefficients, dtype=torch.cfloat, device=device)
        .unsqueeze(0)
        .repeat(int(shots), 1)
    )

    batch_prev: Optional[Tensor] = None

    for i, curr_ten in enumerate(tensor):
        curr_ten = curr_ten.contiguous()

        if i == 0:
            qubit_0 = torch.einsum('ijk,kl->ijl', curr_ten, q0).squeeze(-1)
            qubit_1 = torch.einsum('ijk,kl->ijl', curr_ten, q1).squeeze(-1)
            batch_qubit_0 = qubit_0.unsqueeze(0).expand(int(shots), -1, -1).contiguous()
            batch_qubit_1 = qubit_1.unsqueeze(0).expand(int(shots), -1, -1).contiguous()
        else:
            if batch_prev is not None:
                contracted = torch.einsum(
                    'bij,jkl->bikl', batch_prev, curr_ten
                ).contiguous()
            else:
                contracted = torch.tensor([], device=device)
            batch_qubit_0 = torch.einsum('bijk,kl->bijl', contracted, q0).squeeze(-1).contiguous()
            batch_qubit_1 = torch.einsum('bijk,kl->bijl', contracted, q1).squeeze(-1).contiguous()

        prob_0 = torch.einsum('bij,bij->b', batch_qubit_0.conj(), batch_qubit_0).real
        prob_1 = torch.einsum('bij,bij->b', batch_qubit_1.conj(), batch_qubit_1).real
        total = prob_0 + prob_1

        zero_mask = (total == 0)
        prob_0 = torch.where(zero_mask, torch.full_like(prob_0, 0.5), prob_0)
        prob_1 = torch.where(zero_mask, torch.full_like(prob_1, 0.5), prob_1)
        total = prob_0 + prob_1

        p0 = prob_0 / total
        rnd = torch.rand(int(shots), device=device)
        choose_1 = rnd > p0

        batch_prev = torch.where(
            choose_1[:, None, None],
            batch_qubit_1,
            batch_qubit_0,
        )

        mask = paulis_tensor[:, i].unsqueeze(0).expand(int(shots), -1)
        flip_mask = mask & choose_1.unsqueeze(1)
        batch_coefs = torch.where(flip_mask, -batch_coefs, batch_coefs)

    return torch.mean(batch_coefs.sum(dim=1)).real.detach().cpu().item()
