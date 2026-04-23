"""Batched parameter-shift gradient computation.

Supports auto batch size tuning and multi-GPU distribution.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.multiprocessing as mp
from torch import Tensor

from .model import TensorRingModel
from .tensor_ring.state import TensorRingState
from .measure.efficient_contraction import (
    batched_expectation_value as batched_ev_efficient,
)
from .measure.full_contraction import expectation_value as ev_full


def _get_gpu_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _distribute_params(P: int, num_gpus: int, chunk_size: int) -> dict[int, list[tuple[int, int]]]:
    """Distribute P parameters across num_gpus, each split into chunks."""
    base = P // num_gpus
    remainder = P % num_gpus
    gpu_ranges: dict[int, list[tuple[int, int]]] = {}
    offset = 0
    for gpu_id in range(num_gpus):
        count = base + (1 if gpu_id < remainder else 0)
        if count == 0:
            gpu_ranges[gpu_id] = []
            continue
        end = offset + count
        ranges = []
        for s in range(offset, end, chunk_size):
            ranges.append((s, min(s + chunk_size, end)))
        gpu_ranges[gpu_id] = ranges
        offset = end
    return gpu_ranges


def _gpu_worker(
    gpu_id: int,
    ranges: list[tuple[int, int]],
    base: Tensor,
    model: TensorRingModel,
    shift: float,
    grad_shared: Tensor,
):
    """Compute gradient slices on a specific GPU."""
    device = f"cuda:{gpu_id}"

    for start, stop in ranges:
        C = stop - start
        idx = torch.arange(start, stop, device=device)
        arange_C = torch.arange(C, device=device)

        batch = base.to(device).unsqueeze(0).expand(2 * C, -1).clone()
        batch[arange_C, idx] += shift
        batch[C + arange_C, idx] -= shift

        # Build and evaluate on this GPU
        state = TensorRingState(
            model._num_qubits, model.rank, device,
            model.dtype,
        )
        batch_tensor = state.build_batch(model._gate_templates, batch)

        if model._use_efficient:
            evs = batched_ev_efficient(batch_tensor, model._hamiltonian).to(
                torch.float64
            )
        else:
            evs = torch.zeros(2 * C, dtype=torch.float64)
            for i in range(2 * C):
                evs[i] = ev_full(batch_tensor[i], model._hamiltonian)

        grad_shared[start:stop] = 0.5 * (evs[:C] - evs[C:]) / math.sin(shift)
        del batch_tensor
        torch.cuda.empty_cache()


_VALID_BACKENDS = ("auto", "torch", "jax")


def _resolve_backend_pref(backend_arg: str | None) -> str:
    """Resolve the backend preference for a gradient call.

    Precedence: explicit constructor arg > ``QISKIT_TREV_BACKEND`` env
    var > ``"auto"``. The env var is read on each gradient call (cheap,
    and lets notebooks override without rebuilding objects).
    """
    import os
    import warnings

    if backend_arg is not None:
        if backend_arg not in _VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {_VALID_BACKENDS}, got {backend_arg!r}"
            )
        return backend_arg
    env = os.environ.get("QISKIT_TREV_BACKEND", "auto").lower()
    if env not in _VALID_BACKENDS:
        warnings.warn(
            f"QISKIT_TREV_BACKEND={env!r} not in {_VALID_BACKENDS}; "
            "falling back to 'auto'",
            RuntimeWarning,
            stacklevel=2,
        )
        return "auto"
    return env


class BatchParameterShiftGradient:
    """Compute parameter-shift gradients using batched tensor ring evaluation.

    Supports auto batch size tuning, multi-GPU distribution, and optional
    JAX-backend acceleration.

    Args:
        model: TensorRingModel to compute gradients for.
        shift: Parameter shift amount (default pi/2).
        chunk_size: Params per chunk. None = all at once. "auto" = auto-tune on GPU.
        num_gpus: Number of GPUs to use. None = auto-detect. 0 or 1 = single device.
        backend: Force a compute backend.

            - ``None`` (default): honour the ``QISKIT_TREV_BACKEND`` env var, or
              ``"auto"`` if unset.
            - ``"auto"``: dispatch based on the type of ``params`` at call time —
              ``torch.Tensor`` runs the torch path, ``jax.Array`` runs the
              JIT-compiled JAX path.
            - ``"torch"``: always run the torch path, converting jax inputs
              back if needed.
            - ``"jax"``: always run the JIT-compiled JAX path, converting
              torch inputs over. The return type matches the input type, so
              drop-in use in torch training loops works unchanged.
    """

    def __init__(
        self,
        model: TensorRingModel,
        shift: float = math.pi / 2,
        chunk_size: int | str | None = None,
        num_gpus: int | None = None,
        backend: str | None = None,
    ):
        self._model = model
        self._shift = shift
        self._chunk_size_setting = chunk_size
        self._num_gpus = num_gpus
        self._resolved_chunk_size: int | None = None
        # Validate the constructor arg early (the env-var case is validated
        # on each call in _resolve_backend_pref so a user fixing the env var
        # mid-session doesn't have to rebuild).
        if backend is not None and backend not in _VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {_VALID_BACKENDS}, got {backend!r}"
            )
        self._backend_arg = backend

    def _resolve_chunk_size(self, P: int) -> int:
        """Resolve chunk_size, potentially using auto-tuning."""
        if self._resolved_chunk_size is not None:
            return self._resolved_chunk_size

        setting = self._chunk_size_setting
        if setting is None:
            return P
        if isinstance(setting, int):
            return setting
        if setting == "auto":
            device = torch.device(self._model.device_str)
            if device.type == "cuda" and torch.cuda.is_available():
                from .optimization.auto_batch import auto_batch_size

                model = self._model

                def run_fn(bs: int):
                    C = min(bs, P)
                    if C == 0:
                        return
                    base = torch.zeros(1, P, device=device)
                    batch = base.expand(2 * C, -1).clone()
                    state = TensorRingState(
                        model._num_qubits, model.rank, model.device_str, model.dtype
                    )
                    state.build_batch(model._gate_templates, batch)

                self._resolved_chunk_size = auto_batch_size(
                    run_fn, device, min_bs=1, max_bs=min(4096, P)
                )
                return self._resolved_chunk_size
            else:
                return P
        return P

    @torch.no_grad()
    def __call__(self, params):
        """Compute gradient via batched parameter shift.

        The compute backend is selected by (in precedence order) the
        ``backend=`` constructor arg, the ``QISKIT_TREV_BACKEND`` env var,
        or the input type (``"auto"``). See the class docstring for
        details. Return type always matches input type — drop-in in
        torch training loops.

        Args:
            params: (P,) array of parameter values (torch or jax).

        Returns:
            (P,) array of gradients, same backend as ``params``.
        """
        pref = _resolve_backend_pref(self._backend_arg)

        input_module = type(params).__module__
        is_jax_input = input_module.startswith("jax") or input_module.startswith("jaxlib")

        if pref == "auto":
            use_jax = is_jax_input
        else:
            use_jax = pref == "jax"

        # Convert input to the shape the chosen path expects.
        if use_jax and not is_jax_input:
            compute_params = self._to_jax(params)
        elif not use_jax and is_jax_input:
            compute_params = self._to_torch(params)
        else:
            compute_params = params

        if use_jax:
            result = self._compute_single_jax(compute_params)
        else:
            P = len(compute_params)
            if P == 0:
                result = torch.zeros(0, dtype=torch.float64)
            else:
                num_gpus = self._num_gpus
                if num_gpus is None:
                    num_gpus = _get_gpu_count()
                if num_gpus > 1:
                    result = self._compute_multi_gpu(compute_params, num_gpus)
                else:
                    result = self._compute_single(compute_params)

        # Match output type to input type.
        if use_jax and not is_jax_input:
            return self._to_torch(result)
        if not use_jax and is_jax_input:
            return self._to_jax(result)
        return result

    @staticmethod
    def _to_jax(x):
        import jax.numpy as jnp

        if hasattr(x, "detach"):  # torch.Tensor
            return jnp.asarray(x.detach().cpu().numpy())
        return jnp.asarray(x)

    @staticmethod
    def _to_torch(x):
        import numpy as np

        if hasattr(x, "detach"):  # already a torch tensor
            return x
        # Copy so torch.from_numpy doesn't alias jax's non-writable buffer.
        return torch.from_numpy(np.asarray(x).copy())

    def _compute_single(self, params: Tensor) -> Tensor:
        """Single-device gradient computation."""
        P = len(params)
        model = self._model
        shift = self._shift
        denom = 2 * math.sin(shift)
        chunk_size = self._resolve_chunk_size(P)

        grad = torch.zeros(P, dtype=torch.float64)

        for start in range(0, P, chunk_size):
            stop = min(start + chunk_size, P)
            C = stop - start

            base = params.unsqueeze(0).expand(2 * C, -1).clone()
            idx = torch.arange(start, stop)
            arange_C = torch.arange(C)

            base[arange_C, idx] += shift
            base[C + arange_C, idx] -= shift

            state = TensorRingState(
                model._num_qubits, model.rank, model.device_str, model.dtype
            )
            batch_tensor = state.build_batch(model._gate_templates, base)

            if model._use_efficient:
                evs = batched_ev_efficient(batch_tensor, model._hamiltonian).to(
                    torch.float64
                )
            else:
                evs = torch.zeros(2 * C, dtype=torch.float64)
                for i in range(2 * C):
                    evs[i] = ev_full(batch_tensor[i], model._hamiltonian)

            grad[start:stop] = (evs[:C] - evs[C:]) / denom

        return grad

    def _compute_multi_gpu(self, params: Tensor, num_gpus: int) -> Tensor:
        """Multi-GPU gradient computation using threads."""
        P = len(params)
        chunk_size = self._resolve_chunk_size(P)

        grad_shared = torch.zeros(P, dtype=torch.float64).share_memory_()
        gpu_ranges = _distribute_params(P, num_gpus, chunk_size)

        # Use threads (not processes) for simplicity — GIL released during CUDA ops
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id in range(num_gpus):
                if not gpu_ranges[gpu_id]:
                    continue
                f = executor.submit(
                    _gpu_worker, gpu_id, gpu_ranges[gpu_id],
                    params.cpu(), self._model, self._shift, grad_shared,
                )
                futures.append(f)
            for f in futures:
                f.result()  # raises exceptions from workers

        return grad_shared.clone()

    def _compute_single_jax(self, params):
        """JIT-compiled JAX gradient path.

        Fuses build_batch + batched expectation value + shift diff into one
        XLA graph per (model, shift, P) combination. Per plan/14 Step 4 the
        vmap-equivalent batching is expressed directly via the batch axis
        of build_batch, which is simpler than wrapping in :func:`jax.vmap`
        and compiles to the same program.

        **First-call compile** is ~2–3 s on GPU, ~30–50 s on TPU. That cost
        is paid once per ``(model, shift, P)`` key for the life of the
        Python process. For iterative training it amortises after ~100–500
        gradient steps; below that, the torch path is faster on wall time.
        To collapse the compile cost across process restarts, enable JAX's
        on-disk cache once per process via
        :func:`qiskit_trev.backend.enable_compilation_cache`.
        """
        import jax
        import jax.numpy as jnp
        import numpy as np

        from .backend import JAX_BACKEND
        from .measure.efficient_contraction import _batched_expectation_kernel
        from .tensor_ring._state_jax import build_batch_jax

        P = int(params.shape[0])
        if P == 0:
            return jnp.zeros(0, dtype=jnp.float32)

        model = self._model
        shift = self._shift

        cache = getattr(self, "_jax_jit_cache", None)
        if cache is None:
            cache = {}
            self._jax_jit_cache = cache

        cache_key = (id(model), shift, P)
        jit_fn = cache.get(cache_key)

        if jit_fn is None:
            ham = model._hamiltonian
            paulis = jnp.asarray(ham.get_bool_pauli_tensor().numpy())
            coeffs = jnp.asarray(
                np.asarray(ham.coefficients, dtype=np.complex64)
            )
            Z_op = jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex64)
            I_op = jnp.eye(2, dtype=jnp.complex64)

            num_qubits = model._num_qubits
            rank = model.rank
            gates = model._gate_templates
            denom = 2.0 * math.sin(shift)

            def _grad_fn(params_j):
                eye = jnp.eye(P, dtype=params_j.dtype)
                all_shifted = jnp.concatenate(
                    [
                        params_j[None] + shift * eye,
                        params_j[None] - shift * eye,
                    ],
                    axis=0,
                )
                state = build_batch_jax(num_qubits, rank, gates, all_shifted)
                B = 2 * P
                total_init = jnp.zeros(B, dtype=jnp.complex64)
                evs = _batched_expectation_kernel(
                    JAX_BACKEND,
                    state,
                    paulis,
                    coeffs,
                    Z_op,
                    I_op,
                    total_init,
                    None,
                )
                return (evs[:P] - evs[P:]) / denom

            jit_fn = jax.jit(_grad_fn)
            cache[cache_key] = jit_fn

        return jit_fn(params).block_until_ready()
