"""Chunking correctness for QMLModel JAX path.

Guards the OOM-mitigation change — chunked forward must produce the
same values as the non-chunked forward.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from qiskit.circuit import QuantumCircuit

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from qiskit_trev.qml import QMLModel


def _make_circuit(n_qubits: int = 3, n_layers: int = 1):
    qc = QuantumCircuit(n_qubits)
    di: list[int] = []
    ti: list[int] = []
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(0.0, q)
            di.append(idx); idx += 1
        for q in range(n_qubits):
            qc.ry(0.0, q)
            ti.append(idx); idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc, di, ti


@pytest.mark.parametrize("batch_size", [1, 2, 4, 7])
def test_chunked_forward_matches_unchunked(batch_size):
    qc, di, ti = _make_circuit(n_qubits=3, n_layers=1)
    base = QMLModel(qc, di, ti, rank=4, device="cpu", backend="jax")
    chunked = QMLModel(qc, di, ti, rank=4, device="cpu", backend="jax",
                       batch_size=batch_size)

    N = 7  # a size that's deliberately not a multiple of some batch_sizes
    rng = np.random.RandomState(0)
    X = torch.from_numpy(rng.randn(N, 3).astype(np.float32))
    theta = torch.from_numpy((rng.rand(len(ti)) * 6.28).astype(np.float32))

    out_ref = base(X, theta).cpu().numpy()
    out_chunked = chunked(X, theta).cpu().numpy()

    # The math is identical between chunked and non-chunked; numerical
    # differences come purely from XLA tile-layout changes and tf32
    # matmul rounding (Precision.DEFAULT). Empirically ~7e-4 abs /
    # ~1e-3 rel at this scale — tighter than the gauge-inclusive JAX vs
    # torch comparison, because the gauge issue isn't present here.
    np.testing.assert_allclose(out_chunked, out_ref, atol=1e-3, rtol=1e-2)
