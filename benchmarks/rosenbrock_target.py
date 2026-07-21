"""
Rosenbrock target (2D) and its samplers, shared by the
``rosenbrock_{method}_benchmark.py`` runners.

    U(x, y) = (100 (y - x^2)^2 + (1 - x)^2) / 20.

No closed-form exact samples exist, so MMD is not reported (ESS, R-hat,
acceptance, and KSD via the analytic score).
"""
from __future__ import annotations

import torch
from torch import Tensor

from geodesic_toolbox import (
    HMCSampler, ImplicitRHMCSampler, FHMC, FHMC_initial, IdentityCoMetric,
)
from geodesic_toolbox.cometric import RosenbrockSoftAbs, RosenbrockDualRanders
from benchmark_utils import NUTS

STATE_DIM = 2
NAME = "rosenbrock"


def U(z: Tensor) -> Tensor:
    x, y = z[:, 0], z[:, 1]
    return (100 * (y - x ** 2) ** 2 + (1 - x) ** 2) / 20


def target(z: Tensor) -> Tensor:
    return torch.exp(-U(z))


def log_prob(z: Tensor) -> Tensor:
    x, y = z[0], z[1]
    return -(100 * (y - x ** 2) ** 2 + (1 - x) ** 2) / 20


def score(z: Tensor) -> Tensor:
    """grad log p = -grad U, shape (n, 2)."""
    x, y = z[:, 0], z[:, 1]
    dU_dx = (-400 * x * (y - x ** 2) - 2 * (1 - x)) / 20
    dU_dy = (200 * (y - x ** 2)) / 20
    return -torch.stack([dU_dx, dU_dy], dim=1)


def initial_states(n: int, seed: int | None = None) -> Tensor:
    """Overdispersed chain starts around the ridge."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.randn(n, generator=g) * 1.5
    y = torch.randn(n, generator=g) * 3.0 + 2.0
    return torch.stack([x, y], dim=1)


# ── HMC / RHMC subclasses (U from the Rosenbrock potential) ────────────────────

class RosenbrockHMC(HMCSampler):
    def __init__(self, mass: float, l: int, gamma: float, N_run: int, **kw):
        super().__init__(cometric=IdentityCoMetric(coscale=mass, is_diag=False),
                         l=l, gamma=gamma, N_run=N_run, **kw)

    def U(self, z: Tensor) -> Tensor:
        return U(z)


class RosenbrockRHMC(ImplicitRHMCSampler):
    def __init__(self, l: int, N_fx: int, gamma: float, N_run: int,
                 alpha: float = 1.0, **kw):
        super().__init__(cometric=RosenbrockSoftAbs(alpha=alpha),
                         l=l, N_fx=N_fx, gamma=gamma, N_run=N_run, **kw)

    def U(self, z: Tensor) -> Tensor:
        return U(z)


# ── Sampler builder ────────────────────────────────────────────────────────────

def build_sampler(method: str, params: dict, N_run: int):
    if method == "NUTS":
        return NUTS(log_prob, N_run=N_run, burn=params.get("burn", 200),
                    init_step=params.get("init_step", 0.05),
                    max_steps=params.get("max_steps", 1024),
                    accept=params.get("accept", 0.8))
    if method == "HMC":
        return RosenbrockHMC(mass=params["mass"], l=params["l"], gamma=params["gamma"],
                             N_run=N_run)
    if method == "RHMC":
        return RosenbrockRHMC(l=params["l"], N_fx=params.get("N_fx", 25),
                              gamma=params["gamma"], N_run=N_run,
                              alpha=params.get("alpha", 1.0))
    if method == "FHMC":
        cometric = RosenbrockDualRanders(alpha=params.get("alpha", 1.0), beta=params["beta"])
        return FHMC(target, cometric, l=params["l"], N_fx=params.get("N_fx", 8),
                    gamma=params["gamma"], N_run=N_run, reg=params.get("reg", 0.05),
                    method=params.get("method", "picard"),
                    reduced_flip=params.get("reduced_flip", True))
    if method == "FHMC_INITIAL":
        cometric = RosenbrockDualRanders(alpha=params.get("alpha", 1.0), beta=params["beta"])
        return FHMC_initial(target, cometric, l=params["l"], N_fx=params.get("N_fx", 8),
                            gamma=params["gamma"], N_run=N_run,
                            reduced_flip=params.get("reduced_flip", True))
    raise ValueError(f"unknown method {method!r}")
