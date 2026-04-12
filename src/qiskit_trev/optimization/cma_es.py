"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.

Population-based, derivative-free optimizer that maintains a multivariate
Gaussian and adapts its covariance matrix to learn parameter correlations.

Ported from TREV optimization/cma_es.py (real_form_autograd branch).

Reference:
    Hansen, N. & Ostermeier, A. (2001). "Completely derandomized
    self-adaptation in evolution strategies." Evolutionary Computation 9(2).
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor

from ..model import TensorRingModel


class CMAES:
    """CMA-ES optimizer state.

    Args:
        sigma: Initial step size (standard deviation). Typical: 0.5-1.0.
        pop_size: Population size per generation. None = default 4+floor(3*ln(n)).
        eigen_every: Recompute eigendecomposition of C every this many generations.
    """

    def __init__(
        self,
        sigma: float = 0.5,
        pop_size: int | None = None,
        eigen_every: int = 1,
    ):
        self.sigma0 = sigma
        self.pop_size_override = pop_size
        self.eigen_every = eigen_every

    def _init_state(self, n: int, device: torch.device) -> dict:
        """Initialize all CMA-ES internal state variables."""
        lam = self.pop_size_override or (4 + int(3 * math.log(n)))
        mu = lam // 2

        raw_w = torch.tensor(
            [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)],
            device=device, dtype=torch.float64,
        )
        weights = raw_w / raw_w.sum()
        mu_eff = 1.0 / (weights ** 2).sum().item()

        # Step-size adaptation
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        E_chi = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        # Covariance adaptation
        cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(
            1.0 - c1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff),
        )

        p_sigma = torch.zeros(n, device=device, dtype=torch.float64)
        p_c = torch.zeros(n, device=device, dtype=torch.float64)
        C = torch.eye(n, device=device, dtype=torch.float64)
        sigma = self.sigma0

        BD = torch.eye(n, device=device, dtype=torch.float64)
        invsqrtC = torch.eye(n, device=device, dtype=torch.float64)

        return dict(
            n=n, lam=lam, mu=mu, weights=weights, mu_eff=mu_eff,
            c_sigma=c_sigma, d_sigma=d_sigma, E_chi=E_chi,
            cc=cc, c1=c1, c_mu=c_mu,
            mean=None, p_sigma=p_sigma, p_c=p_c, C=C, sigma=sigma,
            BD=BD, invsqrtC=invsqrtC,
            device=device, gen_count=0,
        )

    def _update_eigen(self, s: dict) -> None:
        """Eigendecompose C and cache BD and invsqrtC."""
        D2, B = torch.linalg.eigh(s['C'])
        D = torch.sqrt(torch.clamp(D2, min=1e-20))
        s['BD'] = B * D
        s['invsqrtC'] = (B / D) @ B.T

    def _step(
        self, s: dict, evaluate_fn: Callable[[Tensor], Tensor]
    ) -> tuple[float, Tensor]:
        """Run one CMA-ES generation.

        Args:
            s: CMA-ES state dict.
            evaluate_fn: Maps (lam, n) tensor → (lam,) cost tensor.

        Returns:
            (best_cost, best_params).
        """
        n, lam, mu = s['n'], s['lam'], s['mu']
        device = s['device']

        if s['gen_count'] % self.eigen_every == 0:
            self._update_eigen(s)
        s['gen_count'] += 1

        # Sample population
        z = torch.randn(lam, n, device=device, dtype=torch.float64)
        y = z @ s['BD'].T
        population = s['mean'] + s['sigma'] * y

        # Evaluate
        costs = evaluate_fn(population.float())

        # Sort by cost
        order = torch.argsort(costs)
        y_sel = y[order[:mu]]
        best_params = population[order[0]]

        # Weighted recombination
        y_w = (s['weights'].unsqueeze(1) * y_sel).sum(dim=0)
        s['mean'] = s['mean'] + s['sigma'] * y_w

        # Step-size path
        s['p_sigma'] = (
            (1.0 - s['c_sigma']) * s['p_sigma']
            + math.sqrt(s['c_sigma'] * (2.0 - s['c_sigma']) * s['mu_eff'])
            * (s['invsqrtC'] @ y_w)
        )
        norm_ps = torch.linalg.norm(s['p_sigma']).item()
        s['sigma'] *= math.exp(
            (s['c_sigma'] / s['d_sigma']) * (norm_ps / s['E_chi'] - 1.0)
        )

        # Covariance path
        h_sigma = 1.0 if (
            norm_ps / math.sqrt(1.0 - (1.0 - s['c_sigma']) ** (2 * (s['gen_count'] + 1)))
            < (1.4 + 2.0 / (n + 1.0)) * s['E_chi']
        ) else 0.0

        s['p_c'] = (
            (1.0 - s['cc']) * s['p_c']
            + h_sigma * math.sqrt(s['cc'] * (2.0 - s['cc']) * s['mu_eff'])
            * y_w
        )

        # Covariance matrix update
        rank_one = s['p_c'].unsqueeze(1) @ s['p_c'].unsqueeze(0)
        w_y = s['weights'].sqrt().unsqueeze(1) * y_sel
        rank_mu = w_y.T @ w_y

        s['C'] = (
            (1.0 - s['c1'] - s['c_mu']) * s['C']
            + s['c1'] * rank_one
            + s['c_mu'] * rank_mu
        )

        return costs[order[0]].item(), best_params


@torch.no_grad()
def minimize_cma_es(
    model: TensorRingModel,
    theta0: Tensor,
    cma: CMAES,
    generations: int,
) -> tuple[Tensor, list[float]]:
    """Run CMA-ES optimization on a TensorRingModel.

    Args:
        model: TensorRingModel to optimize.
        theta0: (P,) initial parameter vector.
        cma: CMAES optimizer instance.
        generations: Number of generations to run.

    Returns:
        (theta, exp_values) where theta is the optimized parameters
        and exp_values is the best cost per generation.
    """
    device = torch.device(model.device_str)
    n = theta0.shape[0]

    s = cma._init_state(n, device)
    s['mean'] = theta0.double().to(device)

    exp_values: list[float] = []

    def evaluate_fn(population: Tensor) -> Tensor:
        """Evaluate a population of parameter vectors."""
        lam = population.shape[0]
        costs = torch.zeros(lam, dtype=torch.float32)
        for i in range(lam):
            costs[i] = model(population[i]).item()
        return costs

    for gen in range(generations):
        best_cost, best_params = cma._step(s, evaluate_fn)
        exp_values.append(best_cost)

    theta = s['mean'].float()
    return theta, exp_values
