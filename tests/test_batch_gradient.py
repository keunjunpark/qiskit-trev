"""Tests for batched expectation value and BatchParameterShiftGradient."""

import math
import pytest
import torch

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_trev.model import TensorRingModel
from qiskit_trev.gradient import BatchParameterShiftGradient


class TestBatchParameterShiftGradient:

    def test_gradient_ry_z(self):
        """d/dtheta cos(theta) = -sin(theta)."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        grad_fn = BatchParameterShiftGradient(model)

        theta = torch.tensor([0.7])
        grad = grad_fn(theta)
        assert abs(grad[0].item() + math.sin(0.7)) < 1e-3

    def test_gradient_matches_sequential(self):
        """Batch gradient should match sequential parameter shift."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        qc.cx(0, 1)
        op = SparsePauliOp.from_list([("ZZ", 1.0)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")

        theta = torch.tensor([0.5, 1.0])
        grad_seq = model.parameter_shift_grad(theta)

        grad_fn = BatchParameterShiftGradient(model)
        grad_batch = grad_fn(theta)

        assert torch.allclose(grad_batch, grad_seq, atol=1e-3)

    def test_gradient_shape(self):
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.rx(0.0, 1)
        op = SparsePauliOp.from_list([("ZI", 1.0)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")
        grad_fn = BatchParameterShiftGradient(model)
        grad = grad_fn(torch.tensor([0.3, 0.7]))
        assert grad.shape == (2,)

    def test_gradient_at_zero(self):
        """At theta=0, d/dtheta cos(theta) = 0."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        grad_fn = BatchParameterShiftGradient(model)
        grad = grad_fn(torch.tensor([0.0]))
        assert abs(grad[0].item()) < 1e-3

    def test_chunk_size(self):
        """Chunked gradient should match full batch."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        op = SparsePauliOp.from_list([("ZZ", 1.0)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")

        theta = torch.tensor([0.5, 1.0])
        grad_full = BatchParameterShiftGradient(model, chunk_size=10)(theta)
        grad_chunked = BatchParameterShiftGradient(model, chunk_size=1)(theta)
        assert torch.allclose(grad_full, grad_chunked, atol=1e-4)

    def test_no_params(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        grad_fn = BatchParameterShiftGradient(model)
        grad = grad_fn(torch.tensor([]))
        assert grad.shape == (0,)

    def test_4_params(self):
        """4-parameter circuit gradient."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        qc.cx(0, 1)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        op = SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")

        theta = torch.tensor([0.3, 0.7, 1.1, 0.2])
        grad_seq = model.parameter_shift_grad(theta)
        grad_batch = BatchParameterShiftGradient(model, chunk_size=2)(theta)
        assert torch.allclose(grad_batch, grad_seq, atol=1e-3)


class TestDistributeParams:
    """Tests for _distribute_params utility function (lines 29-44)."""

    def test_even_distribution(self):
        from qiskit_trev.gradient import _distribute_params
        result = _distribute_params(4, 2, 2)
        assert len(result) == 2
        # GPU 0 gets params 0-1, GPU 1 gets params 2-3
        assert result[0] == [(0, 2)]
        assert result[1] == [(2, 4)]

    def test_uneven_distribution(self):
        from qiskit_trev.gradient import _distribute_params
        result = _distribute_params(5, 2, 3)
        # P=5, n_gpus=2, chunk_size=3
        # base=2, remainder=1 → gpu0 gets 3 params, gpu1 gets 2 params
        assert len(result) == 2
        assert result[0]  # should have at least one chunk

    def test_chunk_size_smaller_than_count(self):
        from qiskit_trev.gradient import _distribute_params
        result = _distribute_params(6, 1, 2)
        # 1 GPU, 6 params, chunk_size=2 → 3 chunks
        assert result[0] == [(0, 2), (2, 4), (4, 6)]

    def test_gpu_with_zero_count(self):
        from qiskit_trev.gradient import _distribute_params
        # P < num_gpus: some GPUs get 0 params
        result = _distribute_params(1, 3, 1)
        assert len(result) == 3
        assert result[0] == [(0, 1)]
        assert result[1] == []
        assert result[2] == []


class TestResolvChunkSizeCaching:
    """Tests for _resolve_chunk_size caching behavior (line 114)."""

    def test_caching_when_resolved_chunk_size_preset(self):
        """When _resolved_chunk_size is pre-set, it is returned directly (line 114)."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        grad_fn = BatchParameterShiftGradient(model)

        # Pre-set the cached value
        grad_fn._resolved_chunk_size = 5
        # Should immediately return 5 from cache (line 114)
        result = grad_fn._resolve_chunk_size(10)
        assert result == 5

    def test_auto_chunk_on_cpu_returns_p(self):
        """chunk_size='auto' on CPU returns P (line 144)."""
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        op = SparsePauliOp.from_list([("ZZ", 1.0)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")
        grad_fn = BatchParameterShiftGradient(model, chunk_size="auto")

        chunk = grad_fn._resolve_chunk_size(2)
        # On CPU, "auto" returns P=2
        assert chunk == 2

    def test_unknown_chunk_size_string_returns_p(self):
        """Unrecognized chunk_size string falls through to return P (line 145)."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("Z", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        # Use an unrecognized string - not None, not int, not "auto"
        grad_fn = BatchParameterShiftGradient(model, chunk_size="unknown")
        chunk = grad_fn._resolve_chunk_size(3)
        # Should fall through all conditions and return P=3
        assert chunk == 3


class TestComputeSingleNonZI:
    """Test _compute_single with non-ZI Hamiltonian (line 202)."""

    def test_gradient_x_hamiltonian(self):
        """Non-ZI Hamiltonian uses ev_full path in _compute_single (line 202)."""
        qc = QuantumCircuit(1)
        qc.ry(0.0, 0)
        op = SparsePauliOp.from_list([("X", 1.0)])
        model = TensorRingModel(qc, op, rank=1, device="cpu")
        grad_fn = BatchParameterShiftGradient(model)

        theta = torch.tensor([0.0])
        grad = grad_fn(theta)
        # d/dtheta <X> for RY(theta)|0>
        # <+|X|+> at theta=0: RY(0)|0>=|0>, <X> on |0>=0, d/dtheta at 0 ≠ 0
        assert grad.shape == (1,)
        assert torch.isfinite(grad).all()

    def test_gradient_xy_hamiltonian_matches_sequential(self):
        """XY Hamiltonian batch gradient matches sequential."""
        from qiskit_trev.model import TensorRingModel as M
        qc = QuantumCircuit(2)
        qc.ry(0.0, 0)
        qc.ry(0.0, 1)
        qc.cx(0, 1)
        op = SparsePauliOp.from_list([("XX", 0.5), ("YY", -0.5)])
        model = TensorRingModel(qc, op, rank=4, device="cpu")

        theta = torch.tensor([0.5, 1.0])
        grad_seq = model.parameter_shift_grad(theta)
        grad_batch = BatchParameterShiftGradient(model)(theta)
        assert torch.allclose(grad_batch, grad_seq, atol=1e-3)


class TestGetGpuCount:

    def test_returns_zero_on_cpu(self):
        """_get_gpu_count returns 0 when CUDA is unavailable (line 22)."""
        from qiskit_trev.gradient import _get_gpu_count
        count = _get_gpu_count()
        # Without GPU, should be 0
        assert count == 0

    def test_gpu_count_with_mocked_cuda(self):
        """_get_gpu_count returns device count when CUDA is available (line 24)."""
        from unittest.mock import patch, MagicMock
        from qiskit_trev.gradient import _get_gpu_count

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2):
            count = _get_gpu_count()
            assert count == 2
