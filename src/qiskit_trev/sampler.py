"""TREVSampler: Qiskit BaseSamplerV2 implementation using tensor rings."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from qiskit.primitives import BaseSamplerV2, PrimitiveJob
from qiskit.primitives.containers import (
    PrimitiveResult,
    PubResult,
    DataBin,
    BitArray,
)
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike

from .converter import circuit_to_gate_instructions
from .tensor_ring.state import TensorRingState
from .measure.full_contraction import measure as full_contraction_measure


class TREVSampler(BaseSamplerV2):
    """GPU-accelerated circuit sampler using tensor rings.

    Conforms to Qiskit BaseSamplerV2. Uses full contraction to compute
    exact probabilities, then samples from the multinomial distribution.
    """

    def __init__(
        self,
        *,
        rank: int = 10,
        device: str = "cpu",
        dtype: torch.dtype = torch.cfloat,
        default_shots: int = 1024,
    ):
        self._rank = rank
        self._device = device
        self._dtype = dtype
        self._default_shots = default_shots

    def run(
        self,
        pubs: Iterable[SamplerPubLike],
        *,
        shots: int | None = None,
    ) -> PrimitiveJob:
        if shots is None:
            shots = self._default_shots
        coerced = [SamplerPub.coerce(p, shots) for p in pubs]
        job = PrimitiveJob(self._run_pubs, coerced)
        job._submit()
        return job

    def _run_pubs(self, pubs):
        results = [self._run_pub(pub) for pub in pubs]
        return PrimitiveResult(results)

    def _run_pub(self, pub: SamplerPub) -> PubResult:
        circuit = pub.circuit
        param_values = pub.parameter_values
        shots = pub.shots

        # Bind parameters if needed
        bc_params = param_values.as_array()
        if bc_params.ndim == 0:
            bc_params = bc_params.reshape(1, -1)
        elif bc_params.ndim == 1:
            bc_params = bc_params.reshape(1, -1)

        sorted_params = sorted(circuit.parameters, key=lambda p: p.name)

        if sorted_params:
            bound = circuit.assign_parameters(
                dict(zip(sorted_params, bc_params[0]))
            )
        else:
            bound = circuit

        # Remove measurement gates for simulation
        sim_circuit = bound.remove_final_measurements(inplace=False)

        # Convert and build tensor ring
        gate_instructions, _ = circuit_to_gate_instructions(sim_circuit)
        state = TensorRingState(
            sim_circuit.num_qubits, self._rank, self._device, self._dtype
        )
        tensor = state.build(gate_instructions)

        # Get exact probabilities
        probs = full_contraction_measure(tensor)
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()

        # Sample bitstrings
        num_qubits = sim_circuit.num_qubits
        indices = np.random.choice(len(probs), size=shots, p=probs)

        # Convert indices to bit arrays (big-endian: qubit 0 = MSB)
        bit_strings = np.zeros((shots, num_qubits), dtype=np.uint8)
        for shot_idx, state_idx in enumerate(indices):
            for q in range(num_qubits):
                bit_strings[shot_idx, q] = (state_idx >> (num_qubits - 1 - q)) & 1

        # Pack into bytes for BitArray
        bit_array = BitArray.from_bool_array(bit_strings.astype(bool))

        return PubResult(DataBin(meas=bit_array, shape=bit_array.shape))
