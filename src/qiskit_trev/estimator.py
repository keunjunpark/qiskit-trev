"""TREVEstimator: Qiskit BaseEstimatorV2 implementation using tensor rings."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from qiskit.primitives import BaseEstimatorV2
from qiskit.primitives.containers import PrimitiveResult, PubResult, DataBin
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.primitives import PrimitiveJob

from qiskit.quantum_info import SparsePauliOp

from .converter import circuit_to_gate_instructions, sparse_pauli_op_to_hamiltonian
from .tensor_ring.state import TensorRingState
from .hamiltonian import Hamiltonian
from .measure.efficient_contraction import expectation_value as ev_efficient
from .measure.full_contraction import expectation_value as ev_full


def _observables_to_sparse_pauli_op(obs) -> SparsePauliOp:
    """Convert an ObservablesArray element (dict) to SparsePauliOp."""
    if isinstance(obs, SparsePauliOp):
        return obs
    if isinstance(obs, dict):
        return SparsePauliOp.from_list([(k, v) for k, v in obs.items()])
    raise TypeError(f"Unsupported observable type: {type(obs)}")


class TREVEstimator(BaseEstimatorV2):
    """GPU-accelerated expectation value estimator using tensor rings.

    Conforms to Qiskit BaseEstimatorV2.
    """

    def __init__(
        self,
        *,
        rank: int = 10,
        device: str = "cpu",
        dtype: torch.dtype = torch.cfloat,
    ):
        self._rank = rank
        self._device = device
        self._dtype = dtype

    def run(
        self,
        pubs: Iterable[EstimatorPubLike],
        *,
        precision: float | None = None,
    ) -> PrimitiveJob:
        coerced = [EstimatorPub.coerce(p) for p in pubs]
        job = PrimitiveJob(self._run_pubs, coerced)
        job._submit()
        return job

    def _run_pubs(self, pubs):
        results = [self._run_pub(pub) for pub in pubs]
        return PrimitiveResult(results)

    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        circuit = pub.circuit
        observables = pub.observables
        param_values = pub.parameter_values

        # Convert observable (scalar ObservablesArray → SparsePauliOp → Hamiltonian)
        obs_flat = observables.ravel()
        sp_op = _observables_to_sparse_pauli_op(obs_flat[0])
        hamiltonian = sparse_pauli_op_to_hamiltonian(sp_op)
        use_efficient = hamiltonian.has_only_zi

        # Handle parameter bindings
        bc_params = param_values.as_array()
        if bc_params.ndim == 0:
            bc_params = bc_params.reshape(1, -1)
        elif bc_params.ndim == 1:
            bc_params = bc_params.reshape(1, -1)

        n_bindings = bc_params.shape[0]
        sorted_params = sorted(circuit.parameters, key=lambda p: p.name)
        evs_list = []

        for i in range(n_bindings):
            values = bc_params[i]
            if sorted_params:
                bound = circuit.assign_parameters(
                    dict(zip(sorted_params, values))
                )
            else:
                bound = circuit
            gate_instructions, _ = circuit_to_gate_instructions(bound)
            state = TensorRingState(
                circuit.num_qubits, self._rank, self._device, self._dtype
            )
            tensor = state.build(gate_instructions)

            if use_efficient:
                ev = ev_efficient(tensor, hamiltonian)
            else:
                ev = ev_full(tensor, hamiltonian)
            evs_list.append(ev)

        evs = np.array(evs_list)
        if n_bindings == 1:
            evs = evs[0]  # scalar for single binding
        stds = np.zeros_like(evs)

        return PubResult(DataBin(evs=evs, stds=stds, shape=evs.shape))

    def _bind_params(self, templates, param_values):
        """Substitute parameter values into gate templates."""
        from .tensor_ring.state import GateInstruction

        gates = []
        param_idx = 0
        for tmpl in templates:
            if tmpl.params:
                n_p = len(tmpl.params)
                if param_idx + n_p <= len(param_values):
                    new_params = tuple(float(param_values[param_idx + i]) for i in range(n_p))
                    param_idx += n_p
                else:
                    new_params = tmpl.params
                gates.append(GateInstruction(tmpl.name, tmpl.qubits, new_params))
            else:
                gates.append(tmpl)
        return gates
