"""Tests for QMLModel: PyTorch-style QML compute primitive."""

import math
import pytest
import torch

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_trev.qml import QMLModel
from qiskit_trev.model import TensorRingModel


def _make_circuit(n_qubits=4, n_layers=1):
    """Build a simple RY ansatz. Returns (circuit, data_indices, trainable_indices).

    Layout per layer: RY_data on each qubit, then RY_trainable on each qubit, then CNOTs.
    """
    qc = QuantumCircuit(n_qubits)
    data_indices = []
    trainable_indices = []
    param_idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(0.0, q)
            data_indices.append(param_idx)
            param_idx += 1
        for q in range(n_qubits):
            qc.ry(0.0, q)
            trainable_indices.append(param_idx)
            param_idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc, data_indices, trainable_indices


class TestInit:

    def test_creates_model(self):
        qc, di, ti = _make_circuit(n_qubits=3)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        assert model.n_qubits == 3
        assert model.n_trainable == len(ti)
        assert model.n_data == len(di)

    def test_precomputed_indices(self):
        """Index tensors should be precomputed at init."""
        qc, di, ti = _make_circuit(n_qubits=2)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        assert model._data_idx.shape == (len(di),)
        assert model._train_idx.shape == (len(ti),)


class TestBuildParamBatch:

    def test_shape(self):
        qc, di, ti = _make_circuit(n_qubits=2)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        X = torch.randn(5, 2)
        theta = torch.randn(len(ti))
        pb = model._build_param_batch(X, theta)
        total_params = len(di) + len(ti)
        assert pb.shape == (5, total_params)

    def test_data_placed_correctly(self):
        qc, di, ti = _make_circuit(n_qubits=2)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        theta = torch.tensor([10.0, 20.0])
        pb = model._build_param_batch(X, theta)
        for row in range(2):
            for j, idx in enumerate(di):
                assert pb[row, idx].item() == pytest.approx(X[row, j].item(), abs=1e-5)
        for j, idx in enumerate(ti):
            assert pb[0, idx].item() == pytest.approx(theta[j].item(), abs=1e-5)


class TestForward:

    def test_shape(self):
        """forward returns (n_qubits, N)."""
        qc, di, ti = _make_circuit(n_qubits=3)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        X = torch.randn(5, 3)
        theta = torch.randn(len(ti))
        evs = model(X, theta)
        assert evs.shape == (3, 5)

    def test_matches_individual_models(self):
        """Each row should match TensorRingModel with single-qubit Z."""
        n_qubits = 3
        qc, di, ti = _make_circuit(n_qubits=n_qubits)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")

        X = torch.randn(4, n_qubits)
        theta = torch.randn(len(ti))
        evs = model(X, theta)

        pb = model._build_param_batch(X, theta)
        for q in range(n_qubits):
            pauli = "I" * (n_qubits - 1 - q) + "Z" + "I" * q
            obs = SparsePauliOp.from_list([(pauli, 1.0)])
            tr_model = TensorRingModel(qc, obs, rank=4, device="cpu")
            ref = tr_model.evaluate_batch(pb)
            torch.testing.assert_close(evs[q], ref, atol=1e-5, rtol=1e-5)

    def test_single_sample(self):
        """N=1 edge case."""
        qc, di, ti = _make_circuit(n_qubits=2)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        evs = model(torch.randn(1, 2), torch.randn(len(ti)))
        assert evs.shape == (2, 1)

    def test_values_bounded(self):
        """Qubit Z expectations in [-1, 1]."""
        qc, di, ti = _make_circuit(n_qubits=2)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        evs = model(torch.randn(10, 2), torch.randn(len(ti)))
        assert evs.min() >= -1.0 - 1e-5
        assert evs.max() <= 1.0 + 1e-5


class TestParameterShiftGrad:

    def test_shape(self):
        """Returns (n_qubits, P, N)."""
        qc, di, ti = _make_circuit(n_qubits=2)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        X = torch.randn(4, 2)
        theta = torch.randn(len(ti))
        dZ = model.parameter_shift_grad(X, theta)
        assert dZ.shape == (2, len(ti), 4)

    def test_finite_difference(self):
        """dZ/dtheta should approximate finite difference."""
        n_qubits = 2
        qc, di, ti = _make_circuit(n_qubits=n_qubits)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        X = torch.randn(3, n_qubits)
        theta = torch.randn(len(ti))

        dZ = model.parameter_shift_grad(X, theta)

        eps = 1e-4
        for p in range(len(ti)):
            tp = theta.clone(); tp[p] += eps
            tm = theta.clone(); tm[p] -= eps
            evs_p = model(X, tp)
            evs_m = model(X, tm)
            fd = (evs_p - evs_m) / (2 * eps)  # (Q, N)
            torch.testing.assert_close(
                dZ[:, p, :], fd, atol=0.05, rtol=0.05,
                msg=f"param {p} gradient mismatch"
            )

    def test_single_sample(self):
        """N=1."""
        qc, di, ti = _make_circuit(n_qubits=2)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        dZ = model.parameter_shift_grad(torch.randn(1, 2), torch.randn(len(ti)))
        assert dZ.shape == (2, len(ti), 1)


class TestForwardPopulation:

    def test_shape(self):
        """Returns (n_qubits, pop_size, N)."""
        qc, di, ti = _make_circuit(n_qubits=3)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        X = torch.randn(5, 3)
        pop = torch.randn(8, len(ti))
        evs = model.forward_population(X, pop)
        assert evs.shape == (3, 8, 5)

    def test_matches_sequential(self):
        """Each candidate should match individual forward call."""
        n_qubits = 2
        qc, di, ti = _make_circuit(n_qubits=n_qubits)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        X = torch.randn(3, n_qubits)
        pop = torch.randn(4, len(ti))

        evs = model.forward_population(X, pop)

        for s in range(4):
            ref = model(X, pop[s])  # (Q, N)
            torch.testing.assert_close(
                evs[:, s, :], ref, atol=1e-5, rtol=1e-5,
                msg=f"candidate {s} mismatch"
            )

    def test_single_candidate(self):
        """pop_size=1."""
        qc, di, ti = _make_circuit(n_qubits=2)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        evs = model.forward_population(torch.randn(3, 2), torch.randn(1, len(ti)))
        assert evs.shape == (2, 1, 3)


class TestEndToEndWorkflow:
    """Verify the intended user workflow from the plan works."""

    def test_param_shift_workflow(self):
        """User-side param shift + output layer + loss computation."""
        n_qubits = 2
        N = 5
        C = 3
        qc, di, ti = _make_circuit(n_qubits=n_qubits)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")

        X = torch.randn(N, n_qubits)
        theta = torch.randn(len(ti))
        W = torch.randn(C, n_qubits, dtype=torch.float64)
        b = torch.zeros(C, dtype=torch.float64)
        y = torch.zeros(N, C, dtype=torch.float64)
        y[:, 0] = 1.0

        # Forward
        evs = model(X, theta)  # (Q, N)
        logits = torch.tanh(W @ evs + b.unsqueeze(1))  # (C, N)
        loss = ((logits.T - y) ** 2).mean()
        assert loss.item() >= 0

        # Gradient
        dl = 2 * (logits.T - y).T / (N * C) * (1 - logits ** 2)
        grad_W = dl @ evs.T
        grad_b = dl.sum(1)
        dZ = model.parameter_shift_grad(X, theta)  # (Q, P, N)
        grad_theta = torch.einsum('qn,qpn->p', W.T @ dl, dZ)

        assert grad_theta.shape == (len(ti),)
        assert grad_W.shape == (C, n_qubits)
        assert grad_b.shape == (C,)

    def test_cma_workflow(self):
        """User-side CMA-ES population evaluation workflow."""
        n_qubits = 2
        N = 4
        C = 2
        qc, di, ti = _make_circuit(n_qubits=n_qubits)
        model = QMLModel(qc, di, ti, rank=4, device="cpu")
        P = len(ti)

        X = torch.randn(N, n_qubits)
        y = torch.zeros(N, C, dtype=torch.float64)
        y[:, 0] = 1.0

        pop_size = 6
        pop = torch.randn(pop_size, P + C * n_qubits + C)

        evs = model.forward_population(X, pop[:, :P])  # (Q, pop, N)

        all_W = pop[:, P:P + C * n_qubits].reshape(pop_size, C, n_qubits).double()
        all_b = pop[:, P + C * n_qubits:].double()  # (pop, C)
        logits = torch.tanh(
            torch.bmm(all_W, evs.permute(1, 0, 2).double()) + all_b.unsqueeze(-1)
        )

        costs = ((logits.permute(0, 2, 1) - y.unsqueeze(0)) ** 2).mean(dim=(1, 2))
        assert costs.shape == (pop_size,)
