"""Auto batch size tuning for GPU memory management.

Ported from TREV optimization/gradients/set_batch_size.py.
"""

from __future__ import annotations

from typing import Callable

import torch


def _gpu_free_bytes(device: torch.device) -> int:
    """Return available GPU memory in bytes."""
    if device.type == "cuda":
        free_b, _ = torch.cuda.mem_get_info(device)
        return int(free_b)
    return 0


@torch.no_grad()
def auto_batch_size(
    run_batch_fn: Callable[[int], None],
    device: torch.device,
    *,
    min_bs: int = 1,
    max_bs: int = 65536,
    safety_frac: float = 0.85,
    growth: float = 2.0,
    warmup: int = 2,
) -> int:
    """Find the largest batch size that fits in GPU memory.

    Performs exponential growth search followed by binary search to find
    the optimal batch size without OOM errors.

    Args:
        run_batch_fn: Function that executes a batch of given size.
        device: Torch device to probe.
        min_bs: Minimum batch size.
        max_bs: Maximum batch size to try.
        safety_frac: Fraction of free memory to target (0.85 = use 85%).
        growth: Exponential growth factor.
        warmup: Number of warmup runs per test.

    Returns:
        Largest stable batch size, or min_bs on CPU.
    """
    assert min_bs >= 1

    if isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda" or not torch.cuda.is_available():
        return min_bs

    if device.index is None:
        idx = torch.cuda.current_device()
        device = torch.device(f"cuda:{idx}")
    else:
        idx = device.index

    torch.cuda.set_device(idx)
    torch.cuda.empty_cache()

    free_b = _gpu_free_bytes(device)
    target_free_b = int(free_b * safety_frac)

    def _try(bs: int) -> bool:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                for _ in range(warmup):
                    run_batch_fn(bs)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            return peak < target_free_b
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
                torch.cuda.empty_cache()
                return False
            raise

    # Exponential growth to find upper bound
    bs = min_bs
    last_ok = min_bs if _try(min_bs) else 0
    if last_ok == 0:
        return min_bs

    hi = max_bs
    while bs < max_bs:
        bs_next = min(int(bs * growth), max_bs)
        if bs_next == bs:
            break
        if _try(bs_next):
            last_ok = bs_next
            bs = bs_next
        else:
            hi = bs_next
            break
    else:
        return last_ok

    # Binary search
    lo = last_ok
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _try(mid):
            lo = mid
        else:
            hi = mid

    return lo
