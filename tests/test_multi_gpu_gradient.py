"""Tests for multi-GPU gradient and batched evaluation."""

import math
import pytest
import torch

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_trev.model import TensorRingModel
from qiskit_trev.gradient import BatchParameterShiftGradient


class TestBatchedEvaluation:

    def test_batched_model_eval(self):
        """Model.evaluate_batch should evaluate multiple param sets at once."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")

        params_batch = torch.tensor([[0.0], [math.pi / 2], [math.pi]])
        evs = model.evaluate_batch(params_batch)
        assert evs.shape == (3,)
        assert abs(evs[0].item() - 1.0) < 1e-4      # cos(0)
        assert abs(evs[1].item()) < 1e-4             # cos(pi/2)
        assert abs(evs[2].item() + 1.0) < 1e-4       # cos(pi)

    def test_batched_eval_matches_sequential(self):
        """Batch evaluation should match sequential."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.rx(0.0, 1)
        qc.cx(0, 1)
        op = SparsePauliOp.from_list([("ZZ", 1.0)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")

        params = torch.tensor([[0.5, 1.0], [1.0, 0.5], [0.3, 0.7]])
        evs_batch = model.evaluate_batch(params)

        for i in range(3):
            ev_seq = model(params[i]).item()
            assert abs(evs_batch[i].item() - ev_seq) < 1e-4


class TestBatchGradientChunking:

    def test_chunk_size_1(self):
        """Chunk size 1 should still work."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        grad_fn = BatchParameterShiftGradient(model, chunk_size=1)
        grad = grad_fn(torch.tensor([0.7]))
        assert abs(grad[0].item() + math.sin(0.7)) < 1e-3

    def test_auto_chunk_size(self):
        """With chunk_size='auto' on CPU, should use all params."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        grad_fn = BatchParameterShiftGradient(model, chunk_size="auto")
        grad = grad_fn(torch.tensor([0.7]))
        assert abs(grad[0].item() + math.sin(0.7)) < 1e-3


class TestCMAESBatchEval:

    def test_cmaes_with_batch_size(self):
        """CMAESOptimizer should accept eval_batch_size."""
        from qiskit_trev.optimization import CMAESOptimizer

        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")

        opt = CMAESOptimizer(sigma=0.5, pop_size=8, eval_batch_size=4)
        result = opt.minimize(model, torch.tensor([0.5]), max_iter=10)
        assert len(result.cost_history) == 10

    def test_cmaes_batch_matches_sequential(self):
        """Batched CMA-ES should converge similarly to sequential."""
        from qiskit_trev.optimization import CMAESOptimizer

        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")

        # Both should converge
        opt_seq = CMAESOptimizer(sigma=0.5, pop_size=8, eval_batch_size=1)
        opt_batch = CMAESOptimizer(sigma=0.5, pop_size=8, eval_batch_size=8)

        torch.manual_seed(42)
        r_seq = opt_seq.minimize(model, torch.tensor([0.5]), max_iter=20)
        torch.manual_seed(42)
        r_batch = opt_batch.minimize(model, torch.tensor([0.5]), max_iter=20)

        # Both should reduce cost significantly
        assert r_seq.cost < -0.5
        assert r_batch.cost < -0.5


@pytest.mark.gpu
class TestMultiGPU:

    def test_multi_gpu_gradient(self):
        """Multi-GPU gradient should match single-GPU."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cuda")

        theta = torch.tensor([0.7])
        grad_single = BatchParameterShiftGradient(model, num_gpus=1)(theta)
        grad_multi = BatchParameterShiftGradient(model, num_gpus=None)(theta)  # auto-detect
        assert torch.allclose(grad_single, grad_multi, atol=1e-3)
