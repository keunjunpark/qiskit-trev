# JAX backend

qiskit-trev ships an optional JAX-JIT compute path alongside the default
PyTorch one. On a modern NVIDIA GPU the JAX path is **2–9× faster per
gradient step** than PyTorch after warm-up, and wins wall-clock on
training runs of roughly **100 iterations or more**.

This tutorial covers: when to use it, how to turn it on, how to make the
cold-start bearable, and how to keep it out of trouble on big workloads.

---

## Should you use JAX?

| Use JAX if | Stay on PyTorch if |
|---|---|
| You run VQE / QAOA / QML for hundreds of gradient steps | You run one-shot / interactive scripts (< 50 iters) |
| You have an NVIDIA GPU (any tier, from RTX 4070 up) | You're on CPU-only hardware |
| Your ansatz is hardware-efficient (RY+CNOT), QAOA (RY+RZZ), or chemistry (RZ + CNOT staircase) | Your workflow is mostly sampling / observable estimation |
| You want the Qiskit Estimator → gradient pipeline to be fast | You depend on tf32-sensitive chemistry requiring Precision.HIGHEST |

TPUs are **not** recommended for current research regimes (χ ≤ 12).
See the plan/14 cross-hardware notes for why.

---

## Install

JAX is not a hard dependency of qiskit-trev. Install it yourself once:

```bash
# Local / datacenter NVIDIA GPU
pip install "jax[cuda12]"

# Colab / Kaggle GPU runtimes: usually preinstalled

# CPU-only (for testing / debugging)
pip install jax
```

Verify:

```python
import jax
print(jax.devices(), jax.default_backend())
# -> [CudaDevice(id=0)] gpu
```

---

## Enable the JAX backend

Three equivalent ways, pick one:

```python
# 1. Per-object (most explicit)
from qiskit_trev.gradient import BatchParameterShiftGradient

grad_fn = BatchParameterShiftGradient(model, backend="jax")
g = grad_fn(params)   # returns the same type you passed in
```

```python
# 2. Session-wide env var (good for notebooks / shell scripts)
# $ QISKIT_TREV_BACKEND=jax python train.py
grad_fn = BatchParameterShiftGradient(model)   # honours env var
```

```python
# 3. Pass JAX arrays directly (dispatch = "auto", the default)
import jax.numpy as jnp

grad_fn = BatchParameterShiftGradient(model)
g = grad_fn(jnp.asarray(my_params))   # runs JAX path, returns jax.Array
```

**Return type always matches input type.** Torch in → torch out. JAX in →
JAX out. Drop-in in a torch training loop with no other code changes.

The same `backend=` argument works on `QMLModel`:

```python
from qiskit_trev.qml import QMLModel

qml = QMLModel(circuit, data_indices, trainable_indices,
               rank=8, backend="jax")
evs   = qml(X, theta)                      # forward
grad  = qml.parameter_shift_grad(X, theta) # gradient
```

---

## First-call compile: the cache is essential

JAX compiles your model the first time you run it. Typical cost:

| Hardware | First-call compile |
|---|---|
| RTX 4070 / 4080 | ~1–3 s |
| A100 / H100 | ~1–3 s |
| TPU v5e-1 | ~30–50 s |

In a **single Python process**, every same-shape call after the first
hits an in-memory cache and is sub-millisecond overhead. So for a
straightforward training script, you pay compile once.

Across **process restarts** (notebook kernel restart, script re-run,
Colab reconnect), the in-memory cache is gone. Turn on the on-disk
compilation cache to skip recompilation:

```python
from qiskit_trev.backend import enable_compilation_cache
enable_compilation_cache()    # writes to ~/.cache/qiskit_trev_jax
```

Subsequent processes reload the compiled binary from disk in ~0.3 s.

### Colab: point the cache at Google Drive

Colab VMs reset on every disconnect — `~/.cache` goes with them. Mount
Drive and put the cache there so it survives:

```python
from google.colab import drive
drive.mount('/content/drive')

from qiskit_trev.backend import enable_compilation_cache
enable_compilation_cache('/content/drive/MyDrive/qiskit_trev_cache')
```

The cache is invalidated automatically when JAX / XLA version, GPU
model, or the traced program (new gate, different χ) changes — stale
artifacts don't silently run.

---

## OOM mitigation: chunk sizes

JAX pre-allocates intermediates when it compiles, so an out-of-memory
failure happens at compile-or-first-run time rather than gradually. The
fix is to chunk the work.

### `BatchParameterShiftGradient`

```python
# Explicit chunk — process at most 8 params per kernel
grad_fn = BatchParameterShiftGradient(model, chunk_size=8, backend="jax")

# Auto-tune: probes the *real* JAX kernel to find the largest chunk
# that compiles + runs without OOM. First call takes ~10-20 s extra
# (the probe); subsequent calls are cached per P.
grad_fn = BatchParameterShiftGradient(model, chunk_size="auto", backend="jax")

# Hard ceiling — auto-tune result is clamped. Useful when you already
# know your hardware's limit (e.g. Colab A100 that OOMs at chunk > 8).
grad_fn = BatchParameterShiftGradient(
    model, chunk_size="auto", max_chunk_size=8, backend="jax"
)
```

### `QMLModel`

Two knobs:

```python
qml = QMLModel(circuit, di, ti, rank=8, backend="jax",
               batch_size=64)   # cap on data samples per forward chunk

qml.auto_tune(N_train)        # torch + JAX refinement; ~10-20 s first call
grad = qml.parameter_shift_grad(X_train, theta)
```

`auto_tune` runs a two-stage probe when `backend="jax"`:

1. **Torch probe** (existing) — sets `batch_size` and an initial
   `_ps_chunk` from the torch forward kernel's memory fit.
2. **JAX refinement** (new) — runs the real
   `parameter_shift_grad_jax` at shrinking chunk sizes until compile
   + execution succeed, then applies a 0.75 safety factor. Caches
   the result per training-set size `N`.

If you already know your hardware's ceiling, pass `max_ps_chunk=K` to
the constructor to skip the JAX refinement entirely:

```python
qml = QMLModel(circuit, di, ti, rank=8, backend="jax", max_ps_chunk=8)
qml.auto_tune(N_train)   # still runs torch probe for batch_size,
                         # but clamps _ps_chunk at 8 without probing.
```

---

## Precision & correctness

JAX on NVIDIA GPU uses **tf32-like reduced-mantissa matmul by default**.
Vs the full-fp32 torch path, this produces a ~1e-4 relative drift. That
is the tradeoff for the speed. Two cases matter:

- **Optimisation loops (VQE / QAOA / QML):** not a problem. Gradient
  descent tolerates 1e-4 gradient noise easily, and the
  physics-accuracy floor in any realistic calculation is nowhere near
  that tight.
- **High-precision expectation values (chemistry Born-Oppenheimer
  energies, benchmarking):** use `JAX_BACKEND_HIGHEST`:
  ```python
  # Not typically needed — only when you really need bit-level torch parity
  ```

When in doubt, compare to `backend="torch"` on a small problem first.

---

## Performance notes

- **Break-even:** JAX wins wall-clock on the full training run from
  roughly **50–100 gradient steps** onwards (A100). Below that, PyTorch
  is faster — stick with torch for quick prototyping.
- **Per-step speedup:** 2–9× torch on A100 after warm-up, workload
  dependent.
- **Compile cost scales with parameter count P, not bond dimension χ.**
  Growing χ is free for compile; growing a deeply-parameterised ansatz
  is expensive. See the compile profile bench in `bench/`.

---

## Known limitations

- **TPU is not a good fit at current-scale workloads** (χ ≤ 12). See
  `plan/14` and `plan/15` local notes for the cross-hardware data.
- **Parameterised 2q gates** outside `{ZZ, ZZ_SWAP}` fall back to
  Qiskit's transpiler decomposition — correct but not fast-path.
- **Custom PyTorch ops in the main training loop** (e.g. custom losses
  that require torch autograd over the gradient) pin you to the torch
  path — JAX doesn't interop with torch's autograd graph.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| First call takes forever | Enable `enable_compilation_cache()`. On Colab, mount Drive. |
| `XlaRuntimeError: RESOURCE_EXHAUSTED` | Use `chunk_size="auto"` / `auto_tune` — they probe the real JAX kernel. Pass `max_chunk_size=K` or `max_ps_chunk=K` if you know the ceiling. |
| `No valid config found!` during compile | Same fix as RESOURCE_EXHAUSTED — XLA autotuner ran out of scratch; shrink the chunk. |
| Numbers differ from torch by ~1e-4 | Expected — tf32 matmul. Use HIGHEST precision for bit-parity. |
| `jax.devices()` shows CPU on GPU runtime | `pip install "jax[cuda12]"` — the default `pip install jax` is CPU-only. |
| TPU runtime is slow | Yes. See above. Use GPU for this workload. |
