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
from .gradient import _resolve_backend_pref, _VALID_BACKENDS
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
        backend: Force a compute backend.

            - ``None`` (default): honour ``QISKIT_TREV_BACKEND`` env var, or
              ``"auto"`` if unset.
            - ``"auto"``: dispatch by ``theta`` type — torch → torch path,
              jax.Array → jitted JAX path.
            - ``"torch"``: always torch.
            - ``"jax"``: always the JIT JAX path. Torch inputs are
              converted to jax for compute; outputs are converted back to
              torch so the return type still matches the input type.
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
        backend: str | None = None,
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

        if backend is not None and backend not in _VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {_VALID_BACKENDS}, got {backend!r}"
            )
        self._backend_arg = backend
        self._jax_jit_cache: dict = {}

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

    def auto_tune(self, N: int) -> None:
        """Auto-tune batch sizes for QML workloads.

        QML has a 2D batching problem: data samples × parameter shifts.
        This method probes GPU memory to find:
          1. batch_size — max samples per _measure_all_qubits call
          2. _ps_chunk  — max parameter shifts per gradient call,
                          computed as batch_size // (2 * N)

        Call once after construction with the training set size:
            model.auto_tune(len(X_train))

        Args:
            N: Number of data samples (training set size).
        """
        from .optimization.auto_batch import auto_batch_size as _abs

        dev = torch.device(self.device)
        if dev.type != "cuda":
            self.batch_size = N
            self._ps_chunk = self.n_trainable
            return

        # Step 1: find max batch_size for _measure_all_qubits
        # Use _measure_all_qubits itself as the probe to capture exact memory
        # (including the (Q, B, chi^4) intermediate tensor).
        dummy = torch.zeros(1, self._total_slots, dtype=torch.float32, device=dev)
        old_bs = self.batch_size

        def _probe_bs(bs):
            p = dummy.expand(bs, -1).contiguous()
            self.batch_size = bs  # no internal chunking during probe
            self._measure_all_qubits(p)

        max_total = N * self.n_trainable * 2
        self.batch_size = _abs(
            _probe_bs, dev, min_bs=N, max_bs=max(max_total, N),
            safety_frac=0.85, warmup=1,
        )

        # Step 2: derive _ps_chunk from batch_size
        # Each param-shift chunk processes 2 * chunk * N samples.
        # Max chunk = batch_size // (2 * N), clamped to [1, n_trainable].
        ps_chunk = max(1, min(self.batch_size // (2 * N), self.n_trainable))
        self._ps_chunk_cache = {N: ps_chunk}

        n_chunks = math.ceil(self.n_trainable / ps_chunk)
        print(f"QMLModel.auto_tune(N={N}):")
        print(f"  batch_size = {self.batch_size:,} samples")
        print(f"  _ps_chunk  = {ps_chunk} params/chunk "
              f"({ps_chunk * 2 * N:,} evals/chunk, {n_chunks} chunks/epoch)")

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

        for start in range(0, B, bs):
            stop = min(start + bs, B)
            chunk = params_batch[start:stop]
            Bc = chunk.shape[0]

            state = TensorRingState(Q, self.rank, self.device, self.dtype)
            bt = state.build_batch(self._gate_templates, chunk)

            # Transfer matrices for I and Z observables at site 0. Two
            # algebraic simplifications vs a naive `einsum(A.conj(), Z@A)`:
            # (1) AO_I = I @ A = A, so the I-branch skips the multiply and
            # uses A directly. (2) Z = diag(1, -1) = I - 2|1><1|, so
            # E_Z = E_I - 2 * (A.conj[...,1] ⊗ A[...,1]) — the correction is
            # half the flops of the original 2-component E_Z einsum.
            A = bt[:, 0]  # (Bc, chi, chi, 2)
            E_I = torch.einsum('blrd,bLRd->blLrR', A.conj(), A)
            corr = torch.einsum('blr,bLR->blLrR', A.conj()[..., 1], A[..., 1])
            E_Z = E_I - 2 * corr

            # ten[q] = E_Z if q==0, else E_I
            ten = E_I.unsqueeze(0).expand(Q, -1, -1, -1, -1, -1).clone()
            ten[0] = E_Z

            # Contract remaining sites. Instead of broadcasting Ei_I across Q
            # and overwriting slot i with Ei_Z (which materializes the full
            # Q-expanded tensor via expand+clone each step), contract the full
            # Q-wide `ten` against `Ei_I` via broadcast, then overwrite slot i
            # with a 1/Q-sized einsum against `Ei_Z`. Eliminates the per-site
            # clone and cuts the main einsum's read bandwidth by Q.
            for i in range(1, Q):
                A = bt[:, i]
                Ei_I = torch.einsum('blrd,bLRd->blLrR', A.conj(), A)
                corr = torch.einsum('blr,bLR->blLrR', A.conj()[..., 1], A[..., 1])
                Ei_Z = Ei_I - 2 * corr

                ten_i_prev = ten[i]
                ten = torch.einsum('Qbijpq,bpqrs->Qbijrs', ten, Ei_I)
                ten[i] = torch.einsum('bijpq,bpqrs->bijrs', ten_i_prev, Ei_Z)

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
        use_jax, theta_is_jax = self._decide_jax(theta)
        if use_jax:
            return self._forward_jax(X, theta, theta_is_jax=theta_is_jax)
        params = self._build_param_batch(X, theta)
        return self._measure_all_qubits(params)

    @torch.no_grad()
    def parameter_shift_grad(
        self, X: Tensor, theta: Tensor, shift: float = math.pi / 2
    ) -> Tensor:
        """See below. JAX path dispatch inserted at the top."""
        use_jax, theta_is_jax = self._decide_jax(theta)
        if use_jax:
            return self._parameter_shift_grad_jax(
                X, theta, shift=shift, theta_is_jax=theta_is_jax
            )
        return self._parameter_shift_grad_torch(X, theta, shift=shift)

    @torch.no_grad()
    def _parameter_shift_grad_torch(
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

        # ── Auto-tune params_per_chunk, adapting to current N ────────
        # Memory ∝ C × N, so the safe chunk size depends on N.
        # Cache per-N and scale for unseen N values.
        if not hasattr(self, '_ps_chunk_cache'):
            self._ps_chunk_cache: dict[int, int] = {}

        if N not in self._ps_chunk_cache:
            if self._ps_chunk_cache:
                # Scale from nearest known N (memory ∝ C × N)
                ref_N = min(self._ps_chunk_cache, key=lambda k: abs(k - N))
                ref_C = self._ps_chunk_cache[ref_N]
                self._ps_chunk_cache[N] = max(1, min(P, int(ref_C * ref_N / N)))
            else:
                # First call ever — probe GPU
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

                self._ps_chunk_cache[N] = _abs(
                    _probe, dev, min_bs=1, max_bs=P,
                    safety_frac=0.85, warmup=1,
                )

        chunk_size = self._ps_chunk_cache[N]

        # ── Compute gradients in chunks ───────────────────────────────
        grad = torch.zeros(self.n_qubits, P, N, dtype=torch.float64, device=dev)

        # Vectorized shift-add: build a sparse (2C, P_total) offset tensor that's
        # +shift at slot train_idx[p] for row 2c, and -shift at the same slot for
        # row 2c+1; zero elsewhere. Broadcast-add onto base to produce the (2C,
        # N, P_total) param batch in one fused op instead of clone + per-param
        # Python scatter.
        rows_C = torch.arange(chunk_size, device=dev)

        for start in range(0, P, chunk_size):
            stop = min(start + chunk_size, P)
            C = stop - start

            chunk_idx = self._train_idx[start:stop]
            shift_add = torch.zeros(C, 2, base.shape[1], dtype=base.dtype, device=dev)
            rows = rows_C[:C]
            shift_add[rows, 0, chunk_idx] = shift
            shift_add[rows, 1, chunk_idx] = -shift
            shift_add = shift_add.view(2 * C, -1)

            blk = (base.unsqueeze(0) + shift_add.unsqueeze(1)).reshape(2 * C * N, -1)
            evs = self._measure_all_qubits(blk)
            evs = evs.view(self.n_qubits, C, 2, N)
            grad[:, start:stop, :] = (evs[:, :, 0, :] - evs[:, :, 1, :]) / denom

        return grad

    @torch.no_grad()
    def forward_population(self, X: Tensor, pop_thetas: Tensor) -> Tensor:
        """See below. JAX path dispatch inserted at the top."""
        use_jax, theta_is_jax = self._decide_jax(pop_thetas)
        if use_jax:
            return self._forward_population_jax(
                X, pop_thetas, theta_is_jax=theta_is_jax
            )
        return self._forward_population_torch(X, pop_thetas)

    @torch.no_grad()
    def _forward_population_torch(self, X: Tensor, pop_thetas: Tensor) -> Tensor:
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

        # Auto-tune if batch_size not set and on GPU
        if self.batch_size is None and torch.device(self.device).type == "cuda":
            self.auto_tune(ps * N)

        all_evs = self._measure_all_qubits(mega)  # (Q, ps*N)
        return all_evs.view(self.n_qubits, ps, N)
    
    def predict(self, X, theta, W, b):
        """Class predictions (numpy). Convenience for evaluation."""
        evs = self.forward(X, theta)
        return torch.tanh(W.to(self.device) @ evs + b.to(self.device).unsqueeze(1)).argmax(0).cpu().numpy()

    # ----- JAX path ---------------------------------------------------

    def _decide_jax(self, probe) -> tuple[bool, bool]:
        """(use_jax, probe_is_jax). `probe` is theta / pop_thetas."""
        pref = _resolve_backend_pref(self._backend_arg)
        mod = type(probe).__module__
        probe_is_jax = mod.startswith("jax") or mod.startswith("jaxlib")
        if pref == "auto":
            return probe_is_jax, probe_is_jax
        return pref == "jax", probe_is_jax

    @staticmethod
    def _to_jax(x):
        import jax.numpy as jnp

        if hasattr(x, "detach"):  # torch.Tensor
            return jnp.asarray(x.detach().cpu().numpy())
        return jnp.asarray(x)

    @staticmethod
    def _to_torch(x, device: str):
        import numpy as np

        if hasattr(x, "detach"):
            return x.to(device)
        return torch.from_numpy(np.asarray(x).copy()).to(device)

    def _jax_index_arrays(self):
        """Return (data_idx, feat_idx, train_idx) as jnp.int32 arrays."""
        import jax.numpy as jnp

        cache = getattr(self, "_jax_index_cache", None)
        if cache is not None:
            return cache
        data_idx = jnp.asarray(self._data_indices, dtype=jnp.int32)
        feat_idx = jnp.asarray(
            [i % self.n_qubits for i in range(len(self._data_indices))],
            dtype=jnp.int32,
        )
        train_idx = jnp.asarray(self._trainable_indices, dtype=jnp.int32)
        self._jax_index_cache = (data_idx, feat_idx, train_idx)
        return self._jax_index_cache

    def _forward_jax(self, X, theta, *, theta_is_jax: bool):
        import jax

        from ._qml_jax import measure_all_qubits_jax, build_param_batch_jax

        X_j = self._to_jax(X)
        theta_j = theta if theta_is_jax else self._to_jax(theta)
        data_idx, feat_idx, train_idx = self._jax_index_arrays()

        cache_key = ("forward", X_j.shape, theta_j.shape[0])
        jit_fn = self._jax_jit_cache.get(cache_key)
        if jit_fn is None:
            def _fn(X_arg, theta_arg):
                p = build_param_batch_jax(
                    X_arg, theta_arg, self._total_slots,
                    data_idx, feat_idx, train_idx,
                )
                return measure_all_qubits_jax(
                    self.n_qubits, self.rank, self._gate_templates, p
                )

            jit_fn = jax.jit(_fn)
            self._jax_jit_cache[cache_key] = jit_fn

        result = jit_fn(X_j, theta_j)
        if theta_is_jax:
            return result  # caller is in JAX land
        # Match input types: return torch on device. Output is float (real).
        import numpy as np
        return torch.from_numpy(np.asarray(result).copy()).to(self.device).to(torch.float64)

    def _parameter_shift_grad_jax(self, X, theta, *, shift, theta_is_jax):
        import jax

        from ._qml_jax import parameter_shift_grad_jax

        X_j = self._to_jax(X)
        theta_j = theta if theta_is_jax else self._to_jax(theta)
        data_idx, feat_idx, train_idx = self._jax_index_arrays()

        chunk_size = getattr(self, "_ps_chunk_cache", {}).get(X_j.shape[0])
        if chunk_size is None:
            chunk_size = self.n_trainable

        cache_key = (
            "param_shift", X_j.shape, theta_j.shape[0], shift, int(chunk_size)
        )
        jit_fn = self._jax_jit_cache.get(cache_key)
        if jit_fn is None:
            def _fn(X_arg, theta_arg):
                return parameter_shift_grad_jax(
                    self.n_qubits, self.rank, self._gate_templates,
                    X_arg, theta_arg,
                    self._total_slots, data_idx, feat_idx, train_idx,
                    shift, chunk_size=chunk_size,
                )

            jit_fn = jax.jit(_fn)
            self._jax_jit_cache[cache_key] = jit_fn

        result = jit_fn(X_j, theta_j)
        if theta_is_jax:
            return result
        import numpy as np
        return torch.from_numpy(np.asarray(result).copy()).to(self.device).to(torch.float64)

    def _forward_population_jax(self, X, pop_thetas, *, theta_is_jax):
        import jax

        from ._qml_jax import forward_population_jax

        X_j = self._to_jax(X)
        pt_j = pop_thetas if theta_is_jax else self._to_jax(pop_thetas)
        data_idx, feat_idx, train_idx = self._jax_index_arrays()

        cache_key = ("forward_pop", X_j.shape, pt_j.shape)
        jit_fn = self._jax_jit_cache.get(cache_key)
        if jit_fn is None:
            def _fn(X_arg, pt_arg):
                return forward_population_jax(
                    self.n_qubits, self.rank, self._gate_templates,
                    X_arg, pt_arg,
                    self._total_slots, data_idx, feat_idx, train_idx,
                )

            jit_fn = jax.jit(_fn)
            self._jax_jit_cache[cache_key] = jit_fn

        result = jit_fn(X_j, pt_j)
        if theta_is_jax:
            return result
        import numpy as np
        return torch.from_numpy(np.asarray(result).copy()).to(self.device).to(torch.float64)
