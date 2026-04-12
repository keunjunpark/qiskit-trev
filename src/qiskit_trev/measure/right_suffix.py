"""Right suffix sampling: Monte Carlo expectation via precomputed right environments.

Ported from TREV measure/right_suffix_sampling.py (real_form_autograd branch).

Works for both chain and ring (periodic) topology. Supports all Pauli
operators (X, Y, Z, I) via QWC grouping and measurement basis rotations.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..hamiltonian import Hamiltonian, rotate_tensor_for_measurement


def _kron(A: Tensor, B: Tensor) -> Tensor:
    return torch.kron(A, B)


def _E_site(core: Tensor):
    """Build double-layer transfer matrices for a site.

    Args:
        core: (chi, chi, 2) tensor.

    Returns:
        (E, (E0, E1)) where E = E0 + E1, each (chi^2, chi^2).
    """
    B0 = core[:, :, 0]
    B1 = core[:, :, 1]
    E0 = _kron(B0, B0.conj())
    E1 = _kron(B1, B1.conj())
    return (E0 + E1), (E0, E1)


@torch.no_grad()
def _precompute_right_suffix(cores: list[Tensor]):
    """Precompute double-layer transfer matrices and right suffixes.

    Args:
        cores: List of N tensors, each (chi, chi, 2).

    Returns:
        Es: List of (E0, E1) tuples, each (chi^2, chi^2).
        R_suf: List of N right-suffix matrices (chi^2, chi^2).
        d2: chi^2.
        device: Torch device.
        dtype: Complex dtype.
    """
    device = cores[0].device
    dtype = torch.complex128 if torch.is_complex(cores[0]) else torch.float64

    E_list, Es = [], []
    for c in cores:
        c = c.to(dtype)
        Ei, (Ei0, Ei1) = _E_site(c)
        E_list.append(Ei)
        Es.append((Ei0, Ei1))

    n = len(E_list)
    d2 = E_list[0].shape[0]
    I = torch.eye(d2, dtype=E_list[0].dtype, device=device)

    # Build right suffixes via reverse accumulation (left-multiply)
    Rpref_rev = [None] * (n + 1)
    acc = I
    Rpref_rev[0] = acc
    E_rev = E_list[::-1]
    for j in range(1, n + 1):
        acc = E_rev[j - 1] @ acc
        Rpref_rev[j] = acc

    R_suf = [Rpref_rev[n - (i + 1)] for i in range(n)]
    return Es, R_suf, d2, device, dtype


@torch.no_grad()
def expectation_value(
    tensor: Tensor,
    hamiltonian: Hamiltonian,
    shots: int = 10000,
    chunk_size: int = 128,
    seed: int | None = None,
) -> float:
    """Compute <psi|H|psi> via right-suffix Monte Carlo sampling.

    Groups Hamiltonian terms by QWC, rotates tensor ring into the
    measurement basis, then samples bitstrings using double-layer
    right-suffix weights.

    Args:
        tensor: (N, chi, chi, 2) tensor ring.
        hamiltonian: Hamiltonian (any Pauli operators).
        shots: Total number of shots (split across QWC groups).
        chunk_size: Shots per chunk.
        seed: Optional RNG seed.

    Returns:
        Real expectation value estimate.
    """
    device = tensor.device
    cdtype = tensor.dtype
    n = tensor.shape[0]

    groups = hamiltonian.get_qwc_groups()
    op_tensor = hamiltonian.get_pauli_op_tensor().to(device)  # (T, N)
    all_coeffs = hamiltonian.coefficients
    shots_per_group = max(1, shots // len(groups))

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    grand_total = 0.0

    for group in groups:
        idx = group['term_indices']

        # Rotate cores for this group's measurement basis
        rotated = rotate_tensor_for_measurement(tensor, group['basis'])
        rot_cores = [rotated[i] for i in range(n)]

        # Non-I mask and coefficients for group terms
        group_nonI = (op_tensor[idx] != 0).to(device=device, dtype=torch.bool)  # (G, N)
        group_coeffs = torch.tensor(
            [all_coeffs[t] for t in idx], dtype=torch.float64, device=device
        )

        # Precompute double layer and right suffixes
        Es, R_suf, d2, _, _ = _precompute_right_suffix(rot_cores)
        chi = int(d2**0.5)
        chi2 = chi * chi

        Es = [(E0.to(cdtype), E1.to(cdtype)) for (E0, E1) in Es]
        R_suf_cast = [Ri.to(cdtype) for Ri in R_suf]
        # Convert R_suf from Kronecker convention to bilinear form
        R_bl = [
            R_suf_cast[i]
            .view(chi, chi, chi, chi)
            .permute(3, 1, 2, 0)
            .contiguous()
            .reshape(chi2, chi2)
            for i in range(n)
        ]
        del R_suf, R_suf_cast

        A0 = [rot_cores[i][:, :, 0].to(cdtype).contiguous() for i in range(n)]
        A1 = [rot_cores[i][:, :, 1].to(cdtype).contiguous() for i in range(n)]

        total = torch.zeros((), dtype=torch.float64, device=device)
        done = 0

        for s0 in range(0, shots_per_group, chunk_size):
            s1 = min(s0 + chunk_size, shots_per_group)
            B = s1 - s0
            X = torch.eye(chi, dtype=cdtype, device=device).expand(B, chi, chi).clone()
            bits = torch.empty((B, n), dtype=torch.bool, device=device)

            for i in range(n):
                M0 = X @ A0[i]
                M1 = X @ A1[i]

                Ri = R_bl[i]
                v0 = M0.reshape(B, chi2)
                v1 = M1.reshape(B, chi2)
                y0 = torch.matmul(v0, Ri.mT)
                y1 = torch.matmul(v1, Ri.mT)
                w0 = (v0.conj() * y0).sum(-1).real
                w1 = (v1.conj() * y1).sum(-1).real

                den = (w0 + w1).clamp_min(1e-300)
                p0 = (w0 / den).to(torch.float64)
                si = (torch.rand((B,), generator=gen, device=device) >= p0)
                bits[:, i] = si
                X = torch.where(si.view(B, 1, 1), M1, M0)

            # Score using non-I mask
            bf = bits.to(torch.float32)
            cnt = bf @ group_nonI.to(torch.float32).T  # (B, G)
            sgn = torch.where((cnt.remainder_(2.0) > 0.5), -1.0, 1.0).to(torch.float64)
            Eb = sgn @ group_coeffs  # (B,)
            total += Eb.sum()
            done += B

        grand_total += (total / max(1, done)).item()

    return grand_total
