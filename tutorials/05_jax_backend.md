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

Three regimes for picking the chunk — from **fastest setup** to **most
memory-optimal**:

| Regime | Setup cost | When to use |
|---|---|---|
| Trust-me: `max_*` override | ~0 s | You already know your hardware's ceiling |
| Analytical estimate (default) | ~0 s | You don't, but don't want to wait |
| JIT probe (opt-in) | 30–600 s cold | You want the absolute largest chunk that fits |

### `BatchParameterShiftGradient`

```python
# Explicit int — no probe, no estimator, no auto-tune. Fast.
grad_fn = BatchParameterShiftGradient(model, chunk_size=8, backend="jax")

# Analytical default (recommended): closed-form memory estimate from
# problem params. ~instant; never OOMs at autotune_level <= 2.
grad_fn = BatchParameterShiftGradient(model, chunk_size="auto", backend="jax")

# Hard ceiling — short-circuits everything. Use when you've already
# diagnosed the ceiling on your hardware.
grad_fn = BatchParameterShiftGradient(
    model, chunk_size="auto", max_chunk_size=8, backend="jax",
)

# Opt-in binary-search probe (old behaviour). Runs 5–8 real compiles
# before the first training step — can take 30–600 s cold. Max memory
# utilisation at the cost of setup time.
grad_fn = BatchParameterShiftGradient(model, chunk_size="jit", backend="jax")
```

### `QMLModel`

```python
qml = QMLModel(circuit, di, ti, rank=8, backend="jax",
               batch_size=64)          # cap on data samples per forward chunk

# Default analytical path — instant.
qml.auto_tune(N_train)

# Known ceiling — short-circuit all probing.
qml = QMLModel(circuit, di, ti, rank=8, backend="jax", max_ps_chunk=8)
qml.auto_tune(N_train)                 # skips analytical AND jit, uses 8 directly

# Old binary-search probe as explicit opt-in.
qml.auto_tune(N_train, probe="jit")    # 30 s–10 min, most accurate

# Skip JAX-specific refinement entirely, trust the torch probe.
qml.auto_tune(N_train, probe="none")

grad = qml.parameter_shift_grad(X_train, theta)
```

**What the analytical estimator does:** computes the closed-form size
of the ``(Q, 2·chunk·N, χ⁴)`` float32 intermediate inside the JAX
gradient kernel, accounts for XLA double-buffering, applies a 5× slack
for moderate autotune scratch, and divides into available HBM. No JAX
compile runs. Calibrated against T4 / A100 / A100-80 probe data; safe
at ``xla_gpu_autotune_level <= 2`` (which `enable_compilation_cache()`
configures).

**If you're seeing OOM with the analytical estimator**, either your
autotune level is at the XLA default (4) or XLA is holding memory from
a previous session. Fix by either:

1. Calling `enable_compilation_cache()` at process start — sets
   autotune_level=2 and gives the estimator an accurate memory model.
2. Passing `max_ps_chunk=K` / `max_chunk_size=K` with a known-safe
   value (halving the analytical estimate is a safe first guess).

---

## How to: OOM-safe JAX training loops

End-to-end recipes for the two common setups. Both assume you've
already enabled the compile cache once at the top of your session
(`enable_compilation_cache(...)`) and installed `jax[cuda12]` /
`jax[tpu]` as appropriate.

### VQE: `TensorRingModel` + `BatchParameterShiftGradient`

```python
import torch
from qiskit_trev.model import TensorRingModel
from qiskit_trev.gradient import BatchParameterShiftGradient

model = TensorRingModel(qc, observable, rank=12, device="cuda")

# JAX backend + auto-tune = safe on any GPU. First call to grad_fn pays
# ~10–30 s for the probe (once per P); every subsequent call is cached.
grad_fn = BatchParameterShiftGradient(
    model,
    shift=torch.pi / 2,
    chunk_size="auto",    # probe finds a safe chunk
    backend="jax",
)

params = torch.rand(model.num_params) * 2 * torch.pi
lr = 0.01

for step in range(500):
    g = grad_fn(params)             # torch → jax → torch, auto-chunked
    params = params - lr * g
```

**If you already know the ceiling** (e.g. Colab A100 OOMs above
`chunk=8` on your circuit):

```python
grad_fn = BatchParameterShiftGradient(
    model,
    chunk_size="auto",
    max_chunk_size=8,     # hard cap — skip probing entirely if you want
    backend="jax",
)
```

### QML: `QMLModel.parameter_shift_grad`

```python
from qiskit_trev.qml import QMLModel

qml = QMLModel(
    circuit, data_indices, trainable_indices,
    rank=12, device="cuda", backend="jax",
    # Omit max_ps_chunk to let the probe find the right value on its own.
    # Pass max_ps_chunk=K if you already know your ceiling.
)

# One-time calibration. When backend="jax", this runs a torch probe for
# forward batch_size AND a JAX probe for gradient _ps_chunk. Plan on
# 20–60 s for the first call on an A100.
qml.auto_tune(len(X_train))

for epoch in range(num_epochs):
    for X_batch, y_batch in loader:
        g = qml.parameter_shift_grad(X_batch, theta)   # (Q, P, B)
        theta = theta - lr * reduce_over_qubits(g, y_batch, ...)
```

### What the probe actually does (and how long it takes)

When `backend="jax"` and you ask for `chunk_size="auto"` /
`auto_tune(...)`:

1. The torch probe (unchanged) picks an optimistic upper bound based
   on CUDA memory fit at 85%.
2. A JAX-native probe runs the real JIT'd kernel at candidate chunk
   sizes, binary-searching for the largest that compiles without
   `RESOURCE_EXHAUSTED` or `No valid config found`.
3. A **0.75 safety factor** is applied on top — training-time churn
   from multiple shape compiles in flight and XLA cache growth can
   push memory higher than the single-shot probe.
4. The result is cached per `P` (for gradient) / per `N` (for QML),
   so every subsequent call within the session skips the probe.

Expected first-call cost on typical hardware:

| Workload | GPU | First `auto_tune` |
|---|---|---|
| VQE, N=10, rank=10, P=20 | A100 40 GB | ~10 s |
| QML, Q=8, N_train=500, P=20 | A100 40 GB | ~20 s |
| QML, Q=16, N_train=500, P=130 | A100 40 GB | ~45 s |

All subsequent calls are cached — the per-session probe cost
disappears completely after the first gradient call.

### Troubleshooting the probe

| Symptom | Cause / fix |
|---|---|
| Probe takes longer than expected | The cache directory is probably not persistent. On Colab, point `enable_compilation_cache` at Google Drive. |
| Probe returns `1` always | Even `chunk=1` can't compile — something's misconfigured (GPU too small for problem, or XLA plugin mismatch). Try `backend="torch"` to confirm the workload size itself is reasonable. |
| Probe passes but training-time OOM | XLA's real-time memory use can exceed probe-time by a few % under cache churn. Add a ceiling: `max_chunk_size=probe_result // 2`. Or lower the `safety_frac` by monkey-patching the helper to `0.5`. |
| Probe succeeds, first training call recompiles | Expected — probe clears its own compile to free memory. First real call compiles once, gets cached. |

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
