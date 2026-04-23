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

See the [`tutorials/`](tutorials/) directory for Jupyter notebooks:

1. **[Getting Started](tutorials/01_getting_started.ipynb)** — Circuits, sampling, and the TREV backend
2. **[Expectation Values](tutorials/02_expectation_values.ipynb)** — Hamiltonians, estimator, and measurement methods
3. **[VQE Optimization](tutorials/03_vqe_optimization.ipynb)** — Gradient descent and CMA-ES for variational algorithms
4. **[Auto Batch Size](tutorials/04_auto_batch_size.ipynb)** — GPU memory-aware batch size tuning for parameter-shift gradients

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

## Experimental JAX backend

An optional JAX backend lives behind `qiskit_trev.backend`. It dispatches
automatically on the input array type — pass a `jax.Array` to
`batched_expectation_value`, `TensorRingState.build_batch`, or
`BatchParameterShiftGradient(...).__call__` and the JAX path runs
(jit-compiled end-to-end); pass a `torch.Tensor` and the existing
PyTorch path runs unchanged.

On a modern NVIDIA GPU the JAX gradient path is roughly 2–9× faster
**per step** than the PyTorch path once warm. The first call pays an
XLA compile cost (~1–3 s on GPU, ~30–50 s on TPU), so the wall-clock
win only appears after ~100–500 iterations. **For shorter scripts or
interactive development, enable the on-disk compilation cache so
second-and-later Python processes skip compile entirely:**

```python
from qiskit_trev.backend import enable_compilation_cache

enable_compilation_cache()  # writes to ~/.cache/qiskit_trev_jax
```

Pass a path explicitly if you want a different cache location. With
the cache enabled, the first run of a given model/parameter-count still
compiles normally; every subsequent Python process (notebook restart,
repeated script, fresh pytest session) reloads the compiled artifact
from disk in under a second.

JAX is not a hard dependency — the backend module imports it lazily, so
torch-only installs keep working.

### Choosing a backend

By default `BatchParameterShiftGradient` dispatches by input type: pass
a `torch.Tensor` and the torch path runs, pass a `jax.Array` and the
JIT JAX path runs. To force one or the other:

```python
# Per-object:
grad_fn = BatchParameterShiftGradient(model, backend="jax")
grad_fn = BatchParameterShiftGradient(model, backend="torch")
grad_fn = BatchParameterShiftGradient(model, backend="auto")   # default

# Session-wide (env var):
#   QISKIT_TREV_BACKEND=jax python train.py
```

With `backend="jax"` and torch-typed `params`, inputs are converted to
JAX, the jit path runs, and the output is converted back to torch —
drop-in in a torch training loop with no other changes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features.

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT License

## Citation

If you use qiskit-trev in your research, please cite:

```bibtex
@software{qiskit_trev,
  title={qiskit-trev: PyTorch-based Tensor Ring VQA Simulation for Qiskit},
  author={Park, Keunjun},
  url={https://github.com/keunjunpark/qiskit-trev},
}
```

## Acknowledgments

This project builds on [TREV](https://github.com/keunjunpark/TREV).
