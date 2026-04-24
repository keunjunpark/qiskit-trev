# qiskit-trev

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/keunjunpark/qiskit-trev/branch/main/graph/badge.svg)](https://codecov.io/gh/keunjunpark/qiskit-trev)
[![Tests](https://github.com/keunjunpark/qiskit-trev/actions/workflows/test.yml/badge.svg)](https://github.com/keunjunpark/qiskit-trev/actions/workflows/test.yml)
[![Qiskit Ecosystem](https://qisk.it/e-e2beb4ad)](https://qisk.it/e)

**Qiskit TREV** is a GPU-accelerated quantum circuit simulation plugin for Qiskit, built on PyTorch. It provides efficient variational quantum algorithm (VQA) simulation using tensor ring (periodic Matrix Product State) representations, powered by PyTorch's GPU acceleration.

## Features

- **Tensor Ring Architecture**: Efficient quantum state representation using periodic Matrix Product States
- **PyTorch Backend**: GPU acceleration via PyTorch tensors and CUDA
- **Qiskit Integration**: Works seamlessly as a Qiskit plugin with `BackendV2` interface
- **Multiple Measurement Methods**:
  - Full Contraction
  - Perfect Sampling
  - Efficient Contraction
  - Right Suffix Contraction
- **Variational Algorithm Support**: Built-in parameter-shift rule gradient computation
- **Hamiltonian Operations**: Full support for Pauli string Hamiltonians via `SparsePauliOp`

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- [PyTorch](https://pytorch.org/) with CUDA
- Qiskit >= 1.0
- NumPy

## Installation

```bash
pip install qiskit-trev
```

For development:

```bash
git clone https://github.com/keunjunpark/qiskit-trev.git
cd qiskit-trev
pip install -e ".[dev]"
```

## Tutorials

See the [`tutorials/`](tutorials/) directory:

1. **[Getting Started](tutorials/01_getting_started.ipynb)** — Circuits, sampling, and the TREV backend
2. **[Expectation Values](tutorials/02_expectation_values.ipynb)** — Hamiltonians, estimator, and measurement methods
3. **[VQE Optimization](tutorials/03_vqe_optimization.ipynb)** — Gradient descent and CMA-ES for variational algorithms
4. **[Auto Batch Size](tutorials/04_auto_batch_size.ipynb)** — GPU memory-aware batch size tuning for parameter-shift gradients
5. **[JAX backend](tutorials/05_jax_backend.md)** — Enable the JAX-JIT path for 2–9× GPU gradient speedup, persistent compile cache (including Colab Drive), backend toggle, and OOM mitigation

## Architecture

```
qiskit_trev/
├── __init__.py               # Public API
├── backend.py                # TREVBackend (Qiskit BackendV2)
├── estimator.py              # TREVEstimator (Qiskit Estimator primitive)
├── sampler.py                # TREVSampler (Qiskit Sampler primitive)
├── tensor_ring/              # Core tensor ring engine
│   ├── state.py              # Tensor ring state representation
│   ├── contraction.py        # Tensor contraction routines
│   └── gates.py              # Gate-to-tensor decomposition
├── measure/                  # Measurement strategies
│   ├── full_contraction.py
│   ├── perfect_sampling.py
│   ├── efficient_contraction.py
│   └── right_suffix.py
└── transpiler/               # TREV-specific transpiler passes
    └── passes.py
```

## How It Differs from TREV

| | TREV | qiskit-trev |
|---|---|---|
| **Backend** | PyTorch | PyTorch |
| **Interface** | Custom `Circuit` API | Qiskit `BackendV2` / Primitives |
| **Gradients** | Parameter-shift rule | Parameter-shift rule |
| **Ecosystem** | Standalone | Qiskit plugin |
| **Install** | `pip install TREV` | `pip install qiskit-trev` |

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features.

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT License

## Citation

If you use qiskit-trev in your research, please cite **both** the
underlying method (TREV paper) and this implementation:

```bibtex
@unpublished{park2025trev,
  title={{TREV}: Python Library for Efficient Implementations of Variational Quantum Algorithms for Optimization using Tensor Networks},
  author={Park, Keun Jun and Peddireddy, Dheeraj and Aggarwal, Vaneet},
  year={2025},
  note={Submitted to ACM Transactions on Quantum Computing; revised April 2026},
}

@software{qiskit_trev,
  title={qiskit-trev: Tensor Ring VQA Simulation as a Qiskit Plugin},
  author={Park, Keun Jun},
  year={2026},
  version={0.2.1},
  url={https://github.com/keunjunpark/qiskit-trev},
  note={Qiskit plugin built on TREV}
}
```

## Acknowledgments

This project builds on [TREV](https://github.com/keunjunpark/TREV).
