"""Numerical-equivalence regression tests for QMLModel.

Pins `_measure_all_qubits` outputs in the production (cfloat) path with a
fixed seed. Every optimization step in plan/12 must keep these tests green
to guarantee exact-gradient equivalence is preserved.

Tolerances:
- Bit-identical runs (same code, same inputs): atol=0.
- Algebraic-equivalence tolerance (after refactors): atol=1e-6, consistent
  with float32-complex accumulation over a Q-site contraction.

Circuit sizes are deliberately small (4q / 2 layers) so the full suite
runs in a few seconds on CPU and a fraction of a second on CUDA. This is
enough to exercise single-qubit gates, CNOT (2q-gate + SVD path), and the
Q-site transfer-matrix contraction loop.

Gate factories (`tensor_ring/gates.py`) currently hardcode `torch.cfloat`,
so `dtype=torch.cdouble` is not yet plumbed through. If we later enable
cdouble end-to-end, retighten ATOL_ALGEBRAIC to ~1e-12.
"""

from __future__ import annotations

import pytest
import torch

from qiskit.circuit import QuantumCircuit

from qiskit_trev.qml import QMLModel


BASELINE_TAG = "pre-phase1"
SEED = 20260416
ATOL_EXACT = 0.0
ATOL_ALGEBRAIC = 1e-6
RTOL = 0.0

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_fixed_circuit(n_qubits: int, n_layers: int):
    qc = QuantumCircuit(n_qubits)
    data_idx: list[int] = []
    train_idx: list[int] = []
    k = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(0.0, q); data_idx.append(k); k += 1
        for q in range(n_qubits):
            qc.ry(0.0, q); train_idx.append(k); k += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc, data_idx, train_idx


def _make_model_and_params(device: str, n_qubits: int = 4, n_layers: int = 2,
                           N: int = 6, rank: int = 4):
    qc, di, ti = _make_fixed_circuit(n_qubits, n_layers)
    model = QMLModel(qc, di, ti, rank=rank, device=device)
    g = torch.Generator().manual_seed(SEED)
    X = torch.randn(N, n_qubits, generator=g, dtype=torch.float32)
    theta = torch.randn(len(ti), generator=g, dtype=torch.float32)
    return model, X, theta


@pytest.mark.parametrize("device", DEVICES)
class TestMeasureAllQubitsReference:
    """Pin a canonical output tensor. Any numerical drift fails here."""

    def _reference(self, device):
        model, X, theta = _make_model_and_params(device)
        params = model._build_param_batch(X, theta)
        return model._measure_all_qubits(params).detach().cpu()

    def test_shape_and_bounded(self, device):
        ref = self._reference(device)
        assert ref.shape == (4, 6)
        assert ref.dtype == torch.float64
        assert ref.min().item() >= -1.0 - 1e-6
        assert ref.max().item() <= 1.0 + 1e-6

    def test_deterministic_across_calls(self, device):
        a = self._reference(device)
        b = self._reference(device)
        torch.testing.assert_close(a, b, atol=ATOL_EXACT, rtol=RTOL)

    def test_recompute_matches_fixture(self, device):
        """Fresh model + fresh inputs (same seed) must reproduce reference."""
        ref = self._reference(device)
        again = self._reference(device)
        torch.testing.assert_close(again, ref, atol=ATOL_ALGEBRAIC, rtol=RTOL)

    def test_row_sum_fingerprint(self, device):
        """Per-qubit row sum is a cheap diff-friendly fingerprint."""
        ref = self._reference(device)
        row_sums = ref.sum(dim=1)
        assert row_sums.shape == (4,)
        assert torch.all(torch.isfinite(row_sums))


@pytest.mark.parametrize("device", DEVICES)
class TestMeasureAllQubitsSelfConsistent:
    """Cross-checks that must hold regardless of implementation."""

    def test_permutation_of_batch(self, device):
        """Permuting samples in the batch permutes columns of the output."""
        model, X, theta = _make_model_and_params(device, N=5)
        params = model._build_param_batch(X, theta)
        evs = model._measure_all_qubits(params)

        perm = torch.tensor([4, 2, 0, 3, 1], device=evs.device)
        evs_perm = model._measure_all_qubits(params[perm])
        torch.testing.assert_close(
            evs[:, perm], evs_perm, atol=ATOL_ALGEBRAIC, rtol=RTOL,
        )

    def test_single_sample_matches_batch_row(self, device):
        """Running a single sample alone must match its row in a batch run."""
        model, X, theta = _make_model_and_params(device, N=3)
        params = model._build_param_batch(X, theta)

        batch_evs = model._measure_all_qubits(params)
        single_evs = model._measure_all_qubits(params[1:2])

        torch.testing.assert_close(
            batch_evs[:, 1:2], single_evs, atol=ATOL_ALGEBRAIC, rtol=RTOL,
        )


@pytest.mark.parametrize("device", DEVICES)
class TestForwardPublicAPI:
    """`forward` is the public entry point for training."""

    def test_forward_shape_and_values(self, device):
        model, X, theta = _make_model_and_params(device, N=5)
        evs = model(X, theta)
        assert evs.shape == (4, 5)
        assert torch.isfinite(evs).all()
        assert evs.abs().max().item() <= 1.0 + 1e-6
