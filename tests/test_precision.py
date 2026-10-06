"""Double precision must work end to end, and must be more accurate.

Before this fix the complex dtype was hardcoded to ``torch.cfloat`` in 14 places
in ``tensor_ring/gates.py`` plus ``measure/efficient_contraction.py`` and
``tensor_ring/contraction.py``, so a ``cdouble`` request either silently
downcast or raised::

    RuntimeError: expected scalar type ComplexFloat but found ComplexDouble

That mattered because truncated tensor-network gradients are sensitive to
rounding: measured against an exact statevector at 12 qubits / depth 3, fp32
reaches only ~1.8e-05 relative error where fp64 reaches ~2.2e-07, and above the
exact-representation bond dimension fp32 error grows erratically while fp64
stays flat.

The exact statevector is the reference throughout -- comparing two tensor-network
code paths against each other only measures self-consistency, and both can agree
while both are wrong.
"""
from __future__ import annotations

import math

import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit_trev.model import TensorRingModel
from qiskit_trev.tensor_ring import gates as gate_fns
from qiskit_trev.tensor_ring.state import TensorRingState

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _ladder(num_qubits: int, depth: int, angles=None):
    """depth x (RY on every qubit, then a CX ladder)."""
    qc = QuantumCircuit(num_qubits)
    k = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(0.0 if angles is None else float(angles[k]), q)
            k += 1
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
    return qc, k


def _observable(num_qubits: int):
    # Qiskit Pauli strings are little-endian: the RIGHTMOST character is qubit
    # 0. On |q0=1>, <"ZI"> = +1 while <"IZ"> = -1.
    return SparsePauliOp.from_list([("I" * (num_qubits - 1) + "Z", 1.0)])


def _exact_gradient(num_qubits: int, depth: int, theta):
    """Parameter-shift gradient from an untruncated statevector."""
    obs = _observable(num_qubits)
    shift = math.pi / 2

    def ev(vals):
        qc, _ = _ladder(num_qubits, depth, vals)
        return Statevector(qc).expectation_value(obs).real

    out = []
    for i in range(len(theta)):
        hi = [float(x) for x in theta]; hi[i] += shift
        lo = [float(x) for x in theta]; lo[i] -= shift
        out.append((ev(hi) - ev(lo)) / (2 * math.sin(shift)))
    return torch.tensor(out, dtype=torch.float64)


# --------------------------------------------------------------- plumbing
@pytest.mark.parametrize("dtype", [torch.cfloat, torch.cdouble])
def test_gate_matrices_follow_state_dtype(dtype):
    """Constructing a state must set the dtype used to build gate matrices."""
    TensorRingState(4, 4, "cpu", dtype)
    assert gate_fns.I("cpu").dtype == dtype
    assert gate_fns.RY(0.3, "cpu").dtype == dtype
    assert gate_fns.CNOT("cpu").dtype == dtype


@pytest.mark.parametrize("device", DEVICES)
def test_cdouble_state_builds(device):
    """A cdouble model must build a cdouble state rather than raising."""
    qc, _ = _ladder(4, 2)
    model = TensorRingModel(qc, _observable(4), rank=4, device=device,
                            dtype=torch.cdouble)
    gates = model._build_gates(torch.zeros(model._num_params,
                                           dtype=torch.float64))
    state = TensorRingState(4, 4, device, torch.cdouble)
    assert state.build(gates).dtype == torch.cdouble


@pytest.mark.parametrize("device", DEVICES)
def test_forward_runs_in_both_precisions(device):
    qc, num_params = _ladder(6, 2)
    obs = _observable(6)
    torch.manual_seed(0)
    theta = torch.randn(num_params, dtype=torch.float64)
    for dtype in (torch.cfloat, torch.cdouble):
        value = TensorRingModel(qc, obs, rank=4, device=device,
                                dtype=dtype).forward(theta)
        assert torch.isfinite(value).all()


# --------------------------------------------------------------- accuracy
@pytest.mark.parametrize("device", DEVICES)
def test_fp64_gradient_is_more_accurate_than_fp32(device):
    """Against an exact statevector, fp64 must beat fp32 by a wide margin.

    rank=8 is the exact-representation bound for depth 3 (each CX layer at most
    doubles the Schmidt rank across a bond), so neither precision is
    truncation-limited here and the gap is purely rounding.
    """
    num_qubits, depth, rank = 8, 3, 8
    qc, num_params = _ladder(num_qubits, depth)
    obs = _observable(num_qubits)
    torch.manual_seed(0)
    theta = torch.randn(num_params, dtype=torch.float64)
    reference = _exact_gradient(num_qubits, depth, theta)
    scale = reference.abs().max().item()

    errors = {}
    for dtype in (torch.cfloat, torch.cdouble):
        grad = TensorRingModel(qc, obs, rank=rank, device=device,
                               dtype=dtype).parameter_shift_grad(theta)
        errors[dtype] = (grad - reference).abs().max().item() / scale

    assert errors[torch.cdouble] < 1e-9, (
        f"fp64 relative error {errors[torch.cdouble]:.2e} is too large; the "
        f"dtype may still be downcast somewhere in the pipeline"
    )
    assert errors[torch.cdouble] < errors[torch.cfloat] / 100, (
        f"expected fp64 to be >100x more accurate, got "
        f"fp32={errors[torch.cfloat]:.2e} fp64={errors[torch.cdouble]:.2e}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_fp64_accuracy_is_flat_above_the_exact_bound(device):
    """Past the exact bond dimension, extra rank must change nothing in fp64.

    For depth 2 the bound is 4, so rank 4, 8 and 16 all represent the state
    exactly and must give the same gradient. In fp32 they do not -- rounding
    accumulates through the larger SVDs.
    """
    num_qubits, depth = 8, 2
    qc, num_params = _ladder(num_qubits, depth)
    obs = _observable(num_qubits)
    torch.manual_seed(0)
    theta = torch.randn(num_params, dtype=torch.float64)
    reference = _exact_gradient(num_qubits, depth, theta)
    scale = reference.abs().max().item()

    errs = []
    for rank in (4, 8, 16):
        grad = TensorRingModel(qc, obs, rank=rank, device=device,
                               dtype=torch.cdouble).parameter_shift_grad(theta)
        errs.append((grad - reference).abs().max().item() / scale)

    assert max(errs) < 1e-9, f"fp64 errors above the bound: {errs}"
    assert max(errs) <= min(errs) * 10 + 1e-12, (
        f"fp64 accuracy should be flat above the exact bound, got {errs}"
    )
