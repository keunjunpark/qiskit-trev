# This code is part of qiskit-trev.
#
# (C) Copyright Keunjun Park 2026.
#
# This code is licensed under the MIT License.

"""Qiskit TREV: GPU-accelerated tensor ring VQA simulation via PyTorch."""

from .tensor_ring import TensorRingState, GateInstruction
from .model import TensorRingModel
from .estimator import TREVEstimator
from .sampler import TREVSampler
from .hamiltonian import Hamiltonian
from .converter import circuit_to_gate_instructions, sparse_pauli_op_to_hamiltonian
from .gradient import BatchParameterShiftGradient
from .optimization import CMAESOptimizer, GradientOptimizer, OptimizationResult
from .optimization.cma_es import CMAES, minimize_cma_es

__all__ = [
    "TensorRingState",
    "GateInstruction",
    "TensorRingModel",
    "TREVEstimator",
    "TREVSampler",
    "Hamiltonian",
    "BatchParameterShiftGradient",
    "CMAESOptimizer",
    "GradientOptimizer",
    "OptimizationResult",
    "CMAES",
    "minimize_cma_es",
    "circuit_to_gate_instructions",
    "sparse_pauli_op_to_hamiltonian",
]
