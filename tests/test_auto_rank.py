"""`required_rank` must report the bond dimension needed for an exact state.

The bound: a unitary acting entirely on one side of a bond cannot change the
Schmidt rank across it. Only gates that CROSS the bond can, and each at most
doubles it. From a product state,

    rank(bond b) <= min(2^(#2q gates crossing b), 2^|left|, 2^|right|)

so for a depth-d linear CX ladder the requirement is min(2^d, 2^floor(n/2)).

Verified against measurement: at 12 qubits, depth 3, the bound is 8, and the
fp64 gradient error falls to 1.1e-02 at rank 7, drops ~50,000x at rank 8, and is
then unchanged through rank 48.
"""
from __future__ import annotations

import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_trev import required_rank
from qiskit_trev.model import TensorRingModel


def _ladder(num_qubits: int, depth: int, ring: bool = False):
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(0.0, q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        if ring:
            qc.cx(num_qubits - 1, 0)
    return qc


@pytest.mark.parametrize("num_qubits", [8, 10, 12, 16])
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_ladder_bound_is_two_to_the_depth(num_qubits, depth):
    """min(2^depth, 2^floor(n/2)) -- the second term binds for shallow wires."""
    got = required_rank(_ladder(num_qubits, depth))["chi"]
    assert got == min(2 ** depth, 2 ** (num_qubits // 2))


def test_edges_need_less_than_the_middle():
    """The per-bond profile is limited by the smaller side near the edges."""
    report = required_rank(_ladder(12, 3))
    assert report["per_bond"] == [2, 4, 8, 8, 8, 8, 8, 8, 8, 4, 2]
    assert report["chi"] == 8
    assert report["bound_exact"] is True


def test_single_qubit_gates_do_not_raise_the_bound():
    """Local unitaries cannot change the Schmidt rank across any bond."""
    qc = QuantumCircuit(6)
    for _ in range(50):
        for q in range(6):
            qc.ry(0.1, q)
            qc.rz(0.2, q)
    assert required_rank(qc)["chi"] == 1


def test_ring_is_flagged_as_not_tight():
    """A periodic bipartition cuts two bonds, so the chain bound is not tight."""
    report = required_rank(_ladder(12, 3, ring=True))
    assert report["bound_exact"] is False
    assert "RING" in report["note"]


def test_cap_marks_truncation():
    report = required_rank(_ladder(16, 6), cap=32)
    assert report["chi"] == 32
    assert report["capped"] is True
    assert "truncation" in report["note"]


def test_rank_auto_is_opt_in_and_default_is_unchanged():
    qc = _ladder(12, 3)
    obs = SparsePauliOp.from_list([("I" * 11 + "Z", 1.0)])

    default = TensorRingModel(qc, obs, device="cpu")
    assert default.rank == 10                 # library default, untouched
    assert default.rank_source == "explicit"
    assert default.rank_report is None

    explicit = TensorRingModel(qc, obs, rank=32, device="cpu")
    assert explicit.rank == 32
    assert explicit.rank_source == "explicit"

    auto = TensorRingModel(qc, obs, rank="auto", device="cpu")
    assert auto.rank == 8
    assert auto.rank_source == "auto"
    assert auto.rank_report["per_bond"][0] == 2


def test_rank_auto_tracks_depth():
    obs = SparsePauliOp.from_list([("I" * 11 + "Z", 1.0)])
    for depth, expected in ((2, 4), (3, 8), (4, 16)):
        m = TensorRingModel(_ladder(12, depth), obs, rank="auto", device="cpu")
        assert m.rank == expected


def test_bad_rank_string_rejected():
    qc = _ladder(8, 2)
    obs = SparsePauliOp.from_list([("I" * 7 + "Z", 1.0)])
    with pytest.raises(ValueError, match="rank must be an int or 'auto'"):
        TensorRingModel(qc, obs, rank="big", device="cpu")


def test_auto_rank_gives_an_exact_gradient():
    """At the computed bound, fp64 must match an exact statevector."""
    import math
    from qiskit.quantum_info import Statevector

    num_qubits, depth = 8, 3
    obs = SparsePauliOp.from_list([("I" * (num_qubits - 1) + "Z", 1.0)])

    def bound_circuit(vals):
        qc = QuantumCircuit(num_qubits)
        k = 0
        for _ in range(depth):
            for q in range(num_qubits):
                qc.ry(float(vals[k]), q); k += 1
            for q in range(num_qubits - 1):
                qc.cx(q, q + 1)
        return qc

    m = TensorRingModel(_ladder(num_qubits, depth), obs, rank="auto",
                        device="cpu", dtype=torch.cdouble)
    assert m.rank == 8
    torch.manual_seed(0)
    theta = torch.randn(m._num_params, dtype=torch.float64)

    shift = math.pi / 2
    exact = []
    for i in range(len(theta)):
        hi = [float(x) for x in theta]; hi[i] += shift
        lo = [float(x) for x in theta]; lo[i] -= shift
        exact.append((Statevector(bound_circuit(hi)).expectation_value(obs).real
                      - Statevector(bound_circuit(lo)).expectation_value(obs).real)
                     / (2 * math.sin(shift)))
    exact = torch.tensor(exact, dtype=torch.float64)

    grad = m.parameter_shift_grad(theta)
    rel = (grad - exact).abs().max().item() / exact.abs().max().item()
    assert rel < 1e-10, f"relative error at the computed bound: {rel:.2e}"
