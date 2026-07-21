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
    HMCSampler, ImplicitRHMCSampler, FHMC, FHMC_initial, IdentityCoMetric,
)
from geodesic_toolbox.cometric import (
    FunnelSoftAbs, FunnelScore, RandersMetrics, DualRandersMetrics,
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

class FunnelSoftAbsRobust(FunnelSoftAbs):
    """FunnelSoftAbs with a scale-normalized eigendecomposition: at extreme
    states (v ~ -6) the Hessian entries reach ~1e5 and LAPACK eigh can fail to
    converge, aborting a batch. Since lam(H/s) * s = lam(H), decomposing the
    normalized matrix is identical and numerically robust."""

    def forward(self, q: Tensor) -> Tensor:
        H = self._hessian(q)
        eps_reg = 1e-3 * torch.arange(H.shape[-1], device=q.device, dtype=q.dtype)
        H = H + torch.diag(eps_reg).unsqueeze(0)
        s = H.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        lam_n, Phi = torch.linalg.eigh(H / s)
        lam = lam_n * s.squeeze(-1)
        alpha_lam = self.alpha * lam
        cometric_eigs = torch.where(
            lam.abs() > 1e-8,
            torch.tanh(alpha_lam) / lam,
            torch.full_like(lam, self.alpha),
        )
        return torch.einsum("bij,bj,bkj->bik", Phi, cometric_eigs, Phi)


def dual_randers(alpha: float, beta: float) -> DualRandersMetrics:
    cometric = FunnelSoftAbsRobust(DIM, alpha)
    omega = FunnelScore(DIM, alpha, cometric=cometric)
    randers = RandersMetrics(base_cometric=cometric, omega=omega, beta=beta)
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
        return FunnelRHMC(l=params["l"], N_fx=params["N_fx"], gamma=params["gamma"],
                          N_run=N_run, alpha=params.get("alpha", 10 ** 6))
    if method == "FHMC":
        cometric = dual_randers(alpha=params["alpha"], beta=params["beta"])
        return FHMC(target, cometric, l=params["l"], N_fx=params["N_fx"],
                    gamma=params["gamma"], N_run=N_run, reg=params.get("reg", 0.05),
                    method=params.get("method", "picard"),
                    reduced_flip=params.get("reduced_flip", True))
    if method == "FHMC_INITIAL":
        cometric = dual_randers(alpha=params["alpha"], beta=params["beta"])
        return FHMC_initial(target, cometric, l=params["l"], N_fx=params["N_fx"],
                            gamma=params["gamma"], N_run=N_run,
                            reduced_flip=params.get("reduced_flip", True))
    raise ValueError(f"unknown method {method!r}")
