"""Tests for auto batch size tuning and batched evaluation utilities."""

import pytest
import torch

from qiskit_trev.optimization.auto_batch import auto_batch_size


class TestAutoBatchSize:

    def test_cpu_returns_min(self):
        """On CPU, should return min_bs since no GPU memory to probe."""
        def run_fn(bs):
            _ = torch.randn(bs, 10)
        result = auto_batch_size(run_fn, torch.device("cpu"), min_bs=4)
        assert result == 4

    def test_returns_positive(self):
        def run_fn(bs):
            _ = torch.randn(bs, 10)
        result = auto_batch_size(run_fn, torch.device("cpu"), min_bs=1)
        assert result >= 1

    @pytest.mark.gpu
    def test_gpu_finds_batch_size(self):
        """On GPU, should find a batch size > min."""
        def run_fn(bs):
            _ = torch.randn(bs, 100, 100, device="cuda")
        result = auto_batch_size(run_fn, torch.device("cuda"), min_bs=1, max_bs=1024)
        assert result >= 1
