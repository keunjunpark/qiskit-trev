"""Bitstring extraction: greedy argmax via right-suffix double-layer.

Finds the highest-probability computational basis state for a tensor ring
without enumerating all 2^N states. Uses right-suffix precomputation
for correct ring (periodic) topology handling.

Ported from TREV measure/contraction.py argmax_tr_noinv_BE and
measure/right_suffix_sampling.py argmax_bitstring_tr_right_suffix.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .right_suffix import _E_site, _precompute_right_suffix


@torch.no_grad()
def argmax_bitstring(tensor: Tensor, normalize_every: int = 8) -> str:
    """Find the highest-probability bitstring for a tensor ring state.

    Uses greedy sequential selection with precomputed right-suffix
    environments in the double-layer representation. Correct for both
    chain and ring (periodic) topology.

    Big-endian output: site 0 = leftmost character.

    Args:
        tensor: (N, chi, chi, 2) tensor ring.
        normalize_every: Stabilize left environment every N sites.

    Returns:
        Bitstring of length N (e.g., "01101").
    """
    cores = [tensor[i] for i in range(tensor.shape[0])]
    Es, R_suf, d2, device, dtype = _precompute_right_suffix(cores)
    n = len(Es)
    eps = 1e-300

    L = torch.eye(d2, dtype=dtype, device=device)
    bits = torch.zeros(n, dtype=torch.long, device=device)

    for i in range(n):
        Ei0, Ei1 = Es[i]
        T0 = L @ Ei0
        T1 = L @ Ei1
        Ri = R_suf[i]

        w0 = torch.trace(T0 @ Ri).real
        w1 = torch.trace(T1 @ Ri).real
        w0c = torch.clamp(w0, min=0.0)
        w1c = torch.clamp(w1, min=0.0)

        if (w0c == 0) and (w1c == 0):
            si = 0
        else:
            si = 1 if w1c > w0c else 0

        bits[i] = si
        L = T1 if si == 1 else T0

        if normalize_every and (i % normalize_every == 0 and i != 0):
            nL = torch.linalg.norm(L.reshape(-1)).clamp_min(eps)
            L = L / nL

    return ''.join(str(int(b.item())) for b in bits)
