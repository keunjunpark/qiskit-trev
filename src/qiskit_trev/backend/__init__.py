"""Backend abstraction for qiskit-trev tensor-ring kernels.

Scope (plan/14 Step 1): narrow protocol covering only the ops reachable from
`measure.efficient_contraction.batched_expectation_value`. Only a torch
implementation exists today; a JAX implementation lands in Step 2.
"""

from ._protocol import Backend
from ._torch import TORCH_BACKEND, TorchBackend


def get_backend(x=None) -> Backend:
    """Return the backend matching the input array.

    Step 1: always returns the torch backend. Step 2 will route to JaxBackend
    when ``x`` is a ``jax.Array``.
    """
    return TORCH_BACKEND


__all__ = ["Backend", "TorchBackend", "TORCH_BACKEND", "get_backend"]
