# qiskit-trev

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Qiskit TREV** is a GPU-accelerated quantum circuit simulation plugin for Qiskit, built on NVIDIA cuQuantum. It provides efficient variational quantum algorithm (VQA) simulation using tensor ring (periodic Matrix Product State) representations, powered natively by cuStateVec and cuTensorNet instead of PyTorch.

## Features

- **Tensor Ring Architecture**: Efficient quantum state representation using periodic Matrix Product States
- **NVIDIA cuQuantum Backend**: Native GPU acceleration via cuStateVec and cuTensorNet (no PyTorch dependency)
- **Qiskit Integration**: Works seamlessly as a Qiskit plugin with `BackendV2` interface
- **Multiple Measurement Methods**:
  - Full Contraction
  - Perfect Sampling
  - Efficient Contraction
  - Right Suffix Contraction
- **Variational Algorithm Support**: Built-in gradient computation and parameter-shift rules
- **Hamiltonian Operations**: Full support for Pauli string Hamiltonians via `SparsePauliOp`

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- [cuQuantum](https://developer.nvidia.com/cuquantum-sdk) (cuStateVec + cuTensorNet)
- Qiskit >= 1.0
- NumPy
- CuPy

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

## Quick Start

```python
from qiskit.circuit import QuantumCircuit
from qiskit_trev import TREVBackend

# Create a Qiskit circuit
qc = QuantumCircuit(4)
qc.h(0)
qc.rx(0.5, 1)
qc.ry(0.3, 2)
qc.cx(0, 3)

# Run on TREV backend with tensor ring rank
backend = TREVBackend(rank=10, device="cuda")
job = backend.run(qc, shots=10000)
result = job.result()
counts = result.get_counts()
```

### Expectation Values with VQE

```python
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_trev import TREVEstimator

# Parameterized ansatz
theta = [Parameter(f"t{i}") for i in range(4)]
qc = QuantumCircuit(4)
for i in range(4):
    qc.h(i)
    qc.ry(theta[i], i)

# Hamiltonian
hamiltonian = SparsePauliOp.from_list([
    ("ZZII", 1.0),
    ("IZZI", 0.5),
    ("IIZZ", 0.5),
])

# GPU-accelerated expectation value
estimator = TREVEstimator(rank=10, device="cuda")
job = estimator.run([(qc, hamiltonian, [0.1, 0.2, 0.3, 0.4])])
result = job.result()
print(f"Expectation value: {result[0].data.evs}")
```

## Architecture

```
qiskit_trev/
├── __init__.py               # Public API
├── backend.py                # TREVBackend (Qiskit BackendV2)
├── estimator.py              # TREVEstimator (Qiskit Estimator primitive)
├── sampler.py                # TREVSampler (Qiskit Sampler primitive)
├── tensor_ring/              # Core tensor ring engine
│   ├── state.py              # Tensor ring state representation
│   ├── contraction.py        # cuTensorNet contraction routines
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
| **Backend** | PyTorch | NVIDIA cuQuantum (cuStateVec + cuTensorNet) |
| **Interface** | Custom `Circuit` API | Qiskit `BackendV2` / Primitives |
| **Autodiff** | PyTorch autograd | Parameter-shift / cuQuantum adjoint |
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

If you use qiskit-trev in your research, please cite:

```bibtex
@software{qiskit_trev,
  title={qiskit-trev: GPU-Accelerated Tensor Ring VQA Simulation for Qiskit},
  author={Park, Keunjun},
  url={https://github.com/keunjunpark/qiskit-trev},
}
```

## Acknowledgments

This project builds on [TREV](https://github.com/keunjunpark/TREV).
