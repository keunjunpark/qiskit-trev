"""QMLModel: PyTorch-style compute primitive for data-driven QML.

The model only computes the forward pass (qubit Z expectation values).
Loss, optimizer, output layer, and training loop are user code — same
philosophy as nn.Linear.

    model = QMLModel(circuit, data_indices, trainable_indices, rank, device)
    evs = model(X, theta)                            # (Q, N) on GPU
    dZ  = model.parameter_shift_grad(X, theta)       # (Q, P, N) on GPU
    evs = model.forward_population(X, pop_thetas)    # (Q, pop, N) on GPU
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .converter import circuit_to_gate_instructions, sparse_pauli_op_to_hamiltonian
from .tensor_ring.state import TensorRingState
from .hamiltonian import Hamiltonian
from .measure.efficient_contraction import batched_expectation_value


class QMLModel:
    """Data-driven QML model: data + trainable params → qubit Z expectations.

    Args:
        circuit: Parameterized QuantumCircuit.
        data_indices: Which parameter slots receive data features.
        trainable_indices: Which parameter slots are trainable weights.
        rank: Tensor ring bond dimension.
        device: "cpu" or "cuda".
        dtype: Complex dtype for tensor ring.
        batch_size: Chunk size for batched evaluation. None = all at once.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        data_indices: list[int],
        trainable_indices: list[int],
        rank: int = 10,
        device: str = "cpu",
        dtype: torch.dtype = torch.cfloat,
        batch_size: int | None = None,
    ):
        self.n_qubits = circuit.num_qubits
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        self._gate_templates, self._total_slots = circuit_to_gate_instructions(circuit)

        # Precompute index tensors for vectorized param construction
        # n_features = number of unique data features (= n_qubits for data re-uploading)
        # data_indices cycles through features: [0,1,...,Q-1, 0,1,...,Q-1, ...]
        # so n_features = n_qubits, and feat_idx maps each data slot to its feature column
        n_features = self.n_qubits
        self._data_idx = torch.tensor(data_indices, dtype=torch.long, device=device)
        self._feat_idx = torch.tensor(
            [i % n_features for i in range(len(data_indices))],
            dtype=torch.long, device=device,
        )
        self._train_idx = torch.tensor(trainable_indices, dtype=torch.long, device=device)

        self._data_indices = data_indices
        self._trainable_indices = trainable_indices

        # Per-qubit Z Hamiltonians
        # Qiskit little-endian: qubit q = position q from the right
        # "I"*(n-1-q) + "Z" + "I"*q → after converter reversal → Z at site q
        self._qubit_hams: list[Hamiltonian] = []
        for q in range(self.n_qubits):
            pauli = "I" * (self.n_qubits - 1 - q) + "Z" + "I" * q
            obs = SparsePauliOp.from_list([(pauli, 1.0)])
            self._qubit_hams.append(sparse_pauli_op_to_hamiltonian(obs))

    @property
    def n_trainable(self) -> int:
        return len(self._trainable_indices)

    @property
    def n_data(self) -> int:
        return len(self._data_indices)

    def _build_param_batch(self, X: Tensor, theta: Tensor) -> Tensor:
        """Build (N, P_total) parameter tensor — vectorized, no Python loops.

        Args:
            X: (N, n_features) data features.
            theta: (P,) trainable circuit parameters.

        Returns:
            (N, P_total) tensor on device.
        """
        N = X.shape[0]
        dev = torch.device(self.device)
        p = torch.zeros(N, self._total_slots, dtype=torch.float32, device=dev)
        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        p[:, self._data_idx] = Xt[:, self._feat_idx]
        p[:, self._train_idx] = theta.float().to(dev).unsqueeze(0)
        return p

    def _build_mega_batch(self, X: Tensor, pop_thetas: Tensor) -> Tensor:
        """Build (pop*N, P_total) parameter tensor for population — vectorized.

        Args:
            X: (N, n_features) data.
            pop_thetas: (pop_size, P) trainable params per candidate.

        Returns:
            (pop_size*N, P_total) tensor on device.
        """
        nd = X.shape[0]
        ps = pop_thetas.shape[0]
        dev = torch.device(self.device)

        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        pt = pop_thetas.float().to(dev)

        m = torch.zeros(ps * nd, self._total_slots, dtype=torch.float32, device=dev)
        m[:, self._data_idx] = Xt[:, self._feat_idx].repeat(ps, 1)
        m[:, self._train_idx] = pt[:, None, :].expand(-1, nd, -1).reshape(ps * nd, -1)
        return m

    @torch.no_grad()
    def _measure_all_qubits(self, params_batch: Tensor) -> Tensor:
        """Build tensor ring once per chunk, measure all qubit Z observables.

        Fuses all Q qubit observables into one pass by carrying
        (Q, B, chi, chi, chi, chi) through the N-site contraction.

        Args:
            params_batch: (B, P_total) on device.

        Returns:
            (n_qubits, B) on device.
        """
        dev = torch.device(self.device)
        B = params_batch.shape[0]
        Q = self.n_qubits
        bs = self.batch_size if self.batch_size is not None else B

        evs = torch.zeros(Q, B, dtype=torch.float64, device=dev)

        Z_op = torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=dev)
        I_op = torch.eye(2, dtype=self.dtype, device=dev)

        for start in range(0, B, bs):
            stop = min(start + bs, B)
            chunk = params_batch[start:stop]
            Bc = chunk.shape[0]

            state = TensorRingState(Q, self.rank, self.device, self.dtype)
            bt = state.build_batch(self._gate_templates, chunk)

            # Site 0: compute E_I and E_Z transfer matrices
            A = bt[:, 0]  # (Bc, chi, chi, 2)
            AO_I = torch.einsum('blrd,dk->blrk', A, I_op)
            AO_Z = torch.einsum('blrd,dk->blrk', A, Z_op)
            E_I = torch.einsum('blrd,bLRd->blLrR', A.conj(), AO_I)
            E_Z = torch.einsum('blrd,bLRd->blLrR', A.conj(), AO_Z)

            # ten[q] = E_Z if q==0, else E_I
            ten = E_I.unsqueeze(0).expand(Q, -1, -1, -1, -1, -1).clone()
            ten[0] = E_Z

            # Contract remaining sites
            for i in range(1, Q):
                A = bt[:, i]
                AO_I = torch.einsum('blrd,dk->blrk', A, I_op)
                AO_Z = torch.einsum('blrd,dk->blrk', A, Z_op)
                Ei_I = torch.einsum('blrd,bLRd->blLrR', A.conj(), AO_I)
                Ei_Z = torch.einsum('blrd,bLRd->blLrR', A.conj(), AO_Z)

                Ei_all = Ei_I.unsqueeze(0).expand(Q, -1, -1, -1, -1, -1).clone()
                Ei_all[i] = Ei_Z

                ten = torch.einsum('Qbijpq,Qbpqrs->Qbijrs', ten, Ei_all)

            # Close ring: trace over (i=r, j=s)
            evs[:, start:stop] = torch.einsum('Qbijij->Qb', ten).real

        return evs

    def __call__(self, X: Tensor, theta: Tensor) -> Tensor:
        """Compute <Z_q> for each qubit and each data point.

        Args:
            X: (N, n_features) data.
            theta: (P,) trainable circuit params.

        Returns:
            (n_qubits, N) tensor on device.
        """
        return self.forward(X, theta)

    @torch.no_grad()
    def forward(self, X: Tensor, theta: Tensor) -> Tensor:
        """Compute <Z_q> for each qubit and each data point.

        Args:
            X: (N, n_features) data.
            theta: (P,) trainable circuit params.

        Returns:
            (n_qubits, N) tensor on device.
        """
        params = self._build_param_batch(X, theta)
        return self._measure_all_qubits(params)

    @torch.no_grad()
    def parameter_shift_grad(
        self, X: Tensor, theta: Tensor, shift: float = math.pi / 2
    ) -> Tensor:
        """Compute d<Z_q>/dtheta_i for all qubits, params, and data points.

        Follows TREV's auto_batch_size pattern: probe GPU once with the
        actual param-shift workload to find maximum params-per-chunk,
        then stream chunks through GPU at full capacity.

        Args:
            X: (N, n_features) data.
            theta: (P,) trainable params.
            shift: Shift amount (default pi/2).

        Returns:
            (n_qubits, P, N) tensor on device.
        """
        P = self.n_trainable
        N = X.shape[0]
        dev = torch.device(self.device)
        denom = 2 * math.sin(shift)

        # Build base params once — (N, P_total) on GPU
        base = self._build_param_batch(X, theta)

        # ── Auto-tune params_per_chunk (TREV pattern) ─────────────────
        # Probe with actual workload: alloc (2C, N, P_total) + measure
        if not hasattr(self, '_ps_chunk') or self._ps_chunk is None:
            from .optimization.auto_batch import auto_batch_size as _abs

            def _probe(C):
                C = min(C, P)
                if C < 1:
                    return
                blk = base.unsqueeze(0).expand(2 * C, -1, -1).clone()
                for j in range(C):
                    blk[2*j, :, self._train_idx[j]] += shift
                    blk[2*j+1, :, self._train_idx[j]] -= shift
                blk = blk.reshape(2 * C * N, -1)
                self._measure_all_qubits(blk)

            self._ps_chunk = _abs(
                _probe, dev, min_bs=1, max_bs=P,
                safety_frac=0.85, warmup=1,
            )

        chunk_size = self._ps_chunk

        # ── Compute gradients in chunks ───────────────────────────────
        grad = torch.zeros(self.n_qubits, P, N, dtype=torch.float64, device=dev)

        for start in range(0, P, chunk_size):
            stop = min(start + chunk_size, P)
            C = stop - start

            blk = base.unsqueeze(0).expand(2 * C, -1, -1).clone()
            for j, p in enumerate(range(start, stop)):
                blk[2*j, :, self._train_idx[p]] += shift
                blk[2*j+1, :, self._train_idx[p]] -= shift

            blk = blk.reshape(2 * C * N, -1)
            evs = self._measure_all_qubits(blk)
            evs = evs.view(self.n_qubits, C, 2, N)
            grad[:, start:stop, :] = (evs[:, :, 0, :] - evs[:, :, 1, :]) / denom

            del blk, evs

        return grad

    @torch.no_grad()
    def forward_population(self, X: Tensor, pop_thetas: Tensor) -> Tensor:
        """Forward pass for a population of theta vectors.

        Mega-batches all candidates x all data points.

        Args:
            X: (N, n_features) data.
            pop_thetas: (pop_size, P) trainable params per candidate.

        Returns:
            (n_qubits, pop_size, N) tensor on device.
        """
        ps = pop_thetas.shape[0]
        N = X.shape[0]

        mega = self._build_mega_batch(X, pop_thetas)
        all_evs = self._measure_all_qubits(mega)  # (Q, ps*N)
        return all_evs.view(self.n_qubits, ps, N)
