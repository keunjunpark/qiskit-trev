"""Phase 0 profiler for QMLModel._measure_all_qubits.

Produces the baseline time breakdown referenced in plan/12. Runs a
parameter-shift-style workload (forward + parameter_shift_grad) under
torch.profiler and prints per-kernel CUDA times + a human-readable
summary table.

Intended run locations:
- CPU smoke: `.venv/bin/python3 bench/profile_qml.py --device cpu --n-qubits 6`
- A100:     `.venv/bin/python3 bench/profile_qml.py --device cuda --n-qubits 16 --rank 12 --batch 50`

The "baseline" run we commit to the repo is the A100 / 16q / rank 12 case.
Re-run after each optimization step (Phase 1A/B/D, Phase 3, ...) and
compare against the previous baseline.
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext

import torch

from qiskit.circuit import QuantumCircuit

from qiskit_trev.qml import QMLModel


def build_qcnn_like_circuit(n_qubits: int, n_layers: int):
    """RY + CNOT-chain ansatz. Matches the shape of the training workload
    from plan/12 (parameterized RY gates + nearest-neighbor entanglers).
    We don't need bit-identical gate choice here — only the gate *counts*
    and the 2q-gate/1q-gate ratio matter for profiling."""
    qc = QuantumCircuit(n_qubits)
    data_idx: list[int] = []
    train_idx: list[int] = []
    k = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(0.0, q); data_idx.append(k); k += 1
        for q in range(n_qubits):
            qc.ry(0.0, q); train_idx.append(k); k += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc, data_idx, train_idx


def make_inputs(model: QMLModel, n_qubits: int, N: int, device: str, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, n_qubits, generator=g)
    theta = torch.randn(model.n_trainable, generator=g)
    return X, theta


def warmup(model: QMLModel, X: torch.Tensor, theta: torch.Tensor, iters: int = 2):
    for _ in range(iters):
        model(X, theta)
    if torch.cuda.is_available() and model.device == "cuda":
        torch.cuda.synchronize()


def time_block(label: str, fn, sync: bool):
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if sync:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    print(f"  {label:<30s} {dt:9.2f} ms")
    return out, dt


def run_profile(args) -> None:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    sync = (device == "cuda")

    qc, di, ti = build_qcnn_like_circuit(args.n_qubits, args.n_layers)
    model = QMLModel(qc, di, ti, rank=args.rank, device=device,
                     batch_size=args.batch if args.batch > 0 else None)
    X, theta = make_inputs(model, args.n_qubits, args.samples, device)

    print(f"\n[config] n_qubits={args.n_qubits} layers={args.n_layers} "
          f"rank={args.rank} samples={args.samples} batch={args.batch} "
          f"device={device}")
    print(f"[config] n_trainable={model.n_trainable} "
          f"(param-shift evals/epoch = {2 * model.n_trainable})")

    # Warmup — first call triggers JIT, cuBLAS handle init, etc.
    print("\n[warmup]")
    warmup(model, X, theta, iters=args.warmup)

    # Coarse wall-clock timing
    print("\n[wall-clock, best-of-3]")
    fwd_times: list[float] = []
    for _ in range(3):
        _, dt = time_block("forward", lambda: model(X, theta), sync)
        fwd_times.append(dt)
    fwd_ms = min(fwd_times)

    grad_times: list[float] = []
    for _ in range(args.grad_repeats):
        _, dt = time_block(
            "parameter_shift_grad",
            lambda: model.parameter_shift_grad(X, theta),
            sync,
        )
        grad_times.append(dt)
    grad_ms = min(grad_times)

    per_eval_ms = grad_ms / (2 * model.n_trainable)
    print(f"\n[summary]")
    print(f"  forward:               {fwd_ms:9.2f} ms")
    print(f"  param-shift-grad:      {grad_ms:9.2f} ms "
          f"({2 * model.n_trainable} evals, {per_eval_ms:.2f} ms/eval)")
    print(f"  grad / forward ratio:  {grad_ms / fwd_ms:9.2f}x "
          f"(expected ~{2 * model.n_trainable}x for naive param-shift)")

    if not args.torch_profiler:
        return

    print("\n[torch.profiler] capturing forward pass")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        with torch.profiler.record_function("forward"):
            model(X, theta)
        if sync:
            torch.cuda.synchronize()

    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=args.row_limit))

    if args.trace_file:
        prof.export_chrome_trace(args.trace_file)
        print(f"\n[trace] wrote {args.trace_file}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0 profiler for QMLModel")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--n-qubits", type=int, default=6)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--samples", type=int, default=32,
                    help="Training set size N")
    ap.add_argument("--batch", type=int, default=0,
                    help="QMLModel.batch_size (0 = all at once)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--grad-repeats", type=int, default=1,
                    help="Param-shift grad repeats (it's expensive)")
    ap.add_argument("--torch-profiler", action="store_true",
                    help="Run torch.profiler for per-kernel breakdown")
    ap.add_argument("--row-limit", type=int, default=25)
    ap.add_argument("--trace-file", default=None,
                    help="Optional chrome-trace output path (.json)")
    args = ap.parse_args()
    run_profile(args)


if __name__ == "__main__":
    main()
