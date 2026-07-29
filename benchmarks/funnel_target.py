"""
Neal's funnel target (dim = 10, state dim 11) and its samplers, shared by the
``funnel_{method}_benchmark.py`` runners.

    U(v, theta) = v^2/18 + (dim/2) v + exp(-v)/2 * sum theta_i^2,
    i.e. v ~ N(0, 9), theta_i | v ~ N(0, e^v).
"""
from __future__ import annotations

import torch
from torch import Tensor

from geodesic_toolbox import (
    HMCSampler, ImplicitRHMCSampler, ImplicitMidpointRHMCSampler, FHMC,
    FHMC_initial, IdentityCoMetric,
)
from geodesic_toolbox.cometric import (
    FunnelSoftAbs, FunnelScore, FunnelScoreTanh, FunnelSkewness,
    RandersMetrics, DualRandersMetrics,
)
from benchmark_utils import NUTS, funnel_exact_samples

DIM = 10
STATE_DIM = DIM + 1
NAME = "funnel"


def U(z: Tensor) -> Tensor:
    v, theta = z[:, 0], z[:, 1:]
    return v ** 2 / 18 + (DIM / 2) * v + torch.exp(-v) / 2 * (theta ** 2).sum(dim=1)


def target(z: Tensor) -> Tensor:
    return torch.exp(-U(z))


def log_prob(z: Tensor) -> Tensor:
    """Single-state log density for NUTS. z is (state_dim,)."""
    v, theta = z[0], z[1:]
    return -(v ** 2 / 18 + (DIM / 2) * v + torch.exp(-v) / 2 * (theta ** 2).sum())


def score(z: Tensor) -> Tensor:
    """grad log p, shape (n, state_dim)."""
    v, theta = z[:, 0], z[:, 1:]
    ev = torch.exp(-v)
    dU_dv = v / 9 + DIM / 2 - ev / 2 * (theta ** 2).sum(dim=1)
    dU_dth = ev.unsqueeze(1) * theta
    return -torch.cat([dU_dv.unsqueeze(1), dU_dth], dim=1)


def reference_samples(n: int, seed: int | None = None) -> Tensor:
    """Exact funnel draws, used as the MMD reference."""
    return funnel_exact_samples(n, DIM, seed)


def initial_states(n: int, seed: int | None = None) -> Tensor:
    """Overdispersed chain starts (exact draws, so R-hat reflects mixing)."""
    return funnel_exact_samples(n, DIM, seed)


# ── Metric ────────────────────────────────────────────────────────────────────

# FunnelSoftAbsRobust used to re-implement FunnelSoftAbs with a
# scale-normalized eigendecomposition (LAPACK eigh can fail on the funnel's
# wildly-scaled Hessian). That normalization now lives inside
# cometric.softabs_cometric, which FunnelSoftAbs uses, so the subclass is
# redundant; the alias is kept so existing scripts keep importing.
FunnelSoftAbsRobust = FunnelSoftAbs


def dual_randers(alpha: float, beta: float, omega: str = "sigmoid",
                 n0: float = 4.0) -> DualRandersMetrics:
    """omega="sigmoid" is the original FunnelScore; "tanh" is FunnelScoreTanh,
    whose 1-form is position dependent and keeps ||b|| away from the Randers
    degeneracy boundary. The two are NOT comparable at equal beta -- compare at
    matched ||b||."""
    cometric = FunnelSoftAbs(DIM, alpha)
    if omega == "tanh":
        om = FunnelScoreTanh(DIM, alpha, cometric=cometric, n0=n0)
    elif omega == "skewness":
        om = FunnelSkewness(DIM, alpha, cometric=cometric, n0=n0)
    elif omega == "sigmoid":
        om = FunnelScore(DIM, alpha, cometric=cometric)
    else:
        raise ValueError(f"unknown omega {omega!r}")
    randers = RandersMetrics(base_cometric=cometric, omega=om, beta=beta)
    return DualRandersMetrics(randers)


# ── HMC / RHMC subclasses (U from the funnel potential) ────────────────────────

class FunnelHMC(HMCSampler):
    def __init__(self, mass: float, l: int, gamma: float, N_run: int, **kw):
        super().__init__(cometric=IdentityCoMetric(coscale=mass, is_diag=False),
                         l=l, gamma=gamma, N_run=N_run, **kw)

    def U(self, z: Tensor) -> Tensor:
        return U(z)


class FunnelRHMC(ImplicitRHMCSampler):
    def __init__(self, l: int, N_fx: int, gamma: float, N_run: int,
                 alpha: float = 10 ** 6, **kw):
        super().__init__(cometric=FunnelSoftAbs(dim=DIM, alpha=alpha),
                         l=l, N_fx=N_fx, gamma=gamma, N_run=N_run, **kw)

    def U(self, z: Tensor) -> Tensor:
        return U(z)


# ── Sampler builder ────────────────────────────────────────────────────────────

def build_sampler(method: str, params: dict, N_run: int):
    """Construct the sampler for `method` with the fixed `params` config."""
    if method == "NUTS":
        return NUTS(log_prob, N_run=N_run, burn=params.get("burn", 200),
                    init_step=params.get("init_step", 0.05),
                    max_steps=params.get("max_steps", 1024),
                    accept=params.get("accept", 0.8))
    if method == "HMC":
        return FunnelHMC(mass=params["mass"], l=params["l"], gamma=params["gamma"],
                         N_run=N_run)
    if method == "RHMC":
        # threshold_fx defaults to 1e-5 in ImplicitRHMCSampler; forward it so the
        # paper's fixed-point tolerance (1e-6) is reachable from PARAMS.
        return FunnelRHMC(l=params["l"], N_fx=params["N_fx"], gamma=params["gamma"],
                          N_run=N_run, alpha=params.get("alpha", 10 ** 6),
                          threshold_fx=params.get("threshold_fx", 1e-5))
    if method == "RHMC_MIDPOINT":
        return ImplicitMidpointRHMCSampler(
            target, FunnelSoftAbsRobust(dim=DIM, alpha=params.get("alpha", 10 ** 6)),
            l=params["l"], N_fx=params["N_fx"], gamma=params["gamma"], N_run=N_run,
            reduced_flip=params.get("reduced_flip", True),
            threshold_fx=params.get("threshold_fx", 1e-12),
        )
    if method == "FHMC":
        cometric = dual_randers(alpha=params["alpha"], beta=params["beta"],
                                omega=params.get("omega", "sigmoid"),
                                n0=params.get("n0", 4.0))
        return FHMC(target, cometric, l=params["l"], N_fx=params["N_fx"],
                    gamma=params["gamma"], N_run=N_run, reg=params.get("reg", 0.05),
                    method=params.get("method", "picard"),
                    reduced_flip=params.get("reduced_flip", True),
                    jacobian=params.get("jacobian", "exact"),
                    jacobian_mc=params.get("jacobian_mc", 1),
                    russian_roulette=params.get("russian_roulette", 0.5))
    if method == "FHMC_INITIAL":
        cometric = dual_randers(alpha=params["alpha"], beta=params["beta"],
                                omega=params.get("omega", "sigmoid"),
                                n0=params.get("n0", 4.0))
        return FHMC_initial(target, cometric, l=params["l"], N_fx=params["N_fx"],
                            gamma=params["gamma"], N_run=N_run,
                            reduced_flip=params.get("reduced_flip", True))
    raise ValueError(f"unknown method {method!r}")
