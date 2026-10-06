"""Bond dimension a circuit needs for an exact representation.

A unitary acting entirely on one side of a bond cannot change the Schmidt rank
across it. Only gates that CROSS the bond can, and each at most doubles it. So
from a product state:

    rank(bond b) <= min( 2^(#two-qubit gates crossing b), 2^|left|, 2^|right| )
    chi = max over bonds

This is an UPPER bound on the exact representation: chi at this value means no
truncation. The true rank may be lower, so it never under-provisions but can
over-provision.
"""
from __future__ import annotations


def required_rank(circuit, cap: int | None = None) -> dict:
    """Bond dimension needed to represent ``circuit``'s output state exactly.

    Args:
        circuit: a Qiskit ``QuantumCircuit``.
        cap: optional ceiling. If the bound exceeds it the result is clamped
            and ``capped`` is set -- expect truncation error, which higher
            precision does NOT fix.

    Returns:
        ``{"chi", "per_bond", "crossings", "bound_exact", "capped", "note"}``.
        ``bound_exact`` is False for a ring (a periodic bipartition cuts two
        bonds, so the chain bound is not tight) or when capped.

    Example:
        >>> from qiskit_trev import required_rank
        >>> required_rank(qc)["chi"]
        8
    """
    n = circuit.num_qubits
    if n < 2:
        return dict(chi=1, per_bond=[], crossings=[], bound_exact=True,
                    capped=False, note="single qubit")

    crossings = [0] * (n - 1)
    wrap = 0
    for inst in circuit.data:
        qs = [circuit.find_bit(q).index for q in inst.qubits]
        if len(qs) != 2:
            continue
        a, b = sorted(qs)
        if (a, b) == (0, n - 1) and n > 2:
            wrap += 1                       # ring-closing bond
            continue
        for bond in range(a, b):
            crossings[bond] += 1

    per_bond = [min(2 ** c, 2 ** (b + 1), 2 ** (n - b - 1))
                for b, c in enumerate(crossings)]
    chi = max(per_bond) if per_bond else 1

    note = ""
    if wrap:
        note = (f"{wrap} wrap-around gate(s) on (0, {n - 1}): this is a RING. "
                f"A periodic bipartition cuts two bonds, so this chain bound "
                f"is not tight.")
    capped = False
    if cap is not None and chi > cap:
        chi, capped = cap, True
        note = (note + " " if note else "") + (
            f"capped at {cap}: expect truncation error, which higher "
            f"precision does not fix")
    return dict(chi=chi, per_bond=per_bond, crossings=crossings,
                bound_exact=(wrap == 0 and not capped), capped=capped,
                note=note.strip())
