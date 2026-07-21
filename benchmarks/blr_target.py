"""
Bayesian logistic-regression target (Framingham data) and its samplers, shared
by the ``blr_{method}_benchmark.py`` runners.

    U(beta) = -[ beta.(X^T y) - sum softplus(X beta) - 1/(2 var) ||beta||^2 ].

U reaches ~thousands (one term per data point), so exp(-U) under/overflows: the
FHMC / FHMC_initial density API (U = -log target) is bypassed by overriding
``U`` analytically in the BLR subclasses below. No exact samples, so MMD is not
reported (ESS, R-hat, acceptance, KSD).
"""
from __future__ import annotations

import os

import torch
from torch import Tensor
from torch.nn.functional import softplus

from geodesic_toolbox import (
    HMCSampler, ImplicitRHMCSampler, FHMC, FHMC_initial, IdentityCoMetric,
)
from geodesic_toolbox.cometric import BLRDualRanders, BLRSoftAbs
from benchmark_utils import NUTS

NAME = "blr"
VAR = 1.0  # prior variance of the coefficients
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "data", "framingham", "framingham.csv")


def _load_framingham(path: str = _DATA):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path).dropna()
    cols_to_drop = [
        "RANDID", "PERIOD", "DEATH", "ANGINA", "HOSPMI", "MI_FCHD", "STROKE",
        "CVD", "HYPERTEN", "TIME", "TIMECHD", "TIMECVD", "TIMEDTH", "TIMEHYP",
        "TIMEAP", "TIMEMI", "TIMEMIFC", "TIMESTRK",
    ]
    X = df.drop(columns=cols_to_drop + ["ANYCHD"])
    y = df["ANYCHD"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(X_tr), dtype=torch.float64)
    X_te = torch.tensor(scaler.transform(X_te), dtype=torch.float64)
    y_tr = torch.tensor(y_tr.values, dtype=torch.float64)
    y_te = torch.tensor(y_te.values, dtype=torch.float64)
    X_tr = torch.cat([X_tr, torch.ones(X_tr.shape[0], 1)], dim=1)  # bias column
    X_te = torch.cat([X_te, torch.ones(X_te.shape[0], 1)], dim=1)
    return X_tr, X_te, y_tr, y_te


X_train, X_test, y_train, y_test = _load_framingham()
STATE_DIM = X_train.shape[1]  # features + bias


# ── Target ─────────────────────────────────────────────────────────────────────

def U(beta: Tensor) -> Tensor:
    """Negative log posterior, shape (n,). beta is (n, state_dim)."""
    lh = beta @ X_train.T @ y_train - softplus(beta @ X_train.T).sum(dim=1)
    prior = -0.5 / VAR * (beta ** 2).sum(dim=1)
    return -(lh + prior)


def log_prob(beta: Tensor) -> Tensor:
    """Single-state log posterior for NUTS. beta is (state_dim,)."""
    return -U(beta.unsqueeze(0)).squeeze(0)


def score(beta: Tensor) -> Tensor:
    """grad log p = X^T (y - sigmoid(X beta)) - beta / var, shape (n, state_dim)."""
    p = torch.sigmoid(beta @ X_train.T)          # (n, N_data)
    return (y_train - p) @ X_train - beta / VAR


def initial_states(n: int, seed: int | None = None) -> Tensor:
    """Overdispersed chain starts from the (unit-variance) prior."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.randn(n, STATE_DIM, generator=g)


# ── Samplers (U overridden analytically; the target arg is unused) ─────────────

_UNUSED_TARGET = lambda z: torch.ones(z.shape[0], device=z.device, dtype=z.dtype)


class BLRHMC(HMCSampler):
    def __init__(self, mass: float, l: int, gamma: float, N_run: int, **kw):
        super().__init__(cometric=IdentityCoMetric(coscale=mass, is_diag=False),
                         l=l, gamma=gamma, N_run=N_run, **kw)

    def U(self, z: Tensor) -> Tensor:
        return U(z)


class BLRRHMC(ImplicitRHMCSampler):
    def __init__(self, l: int, N_fx: int, gamma: float, N_run: int,
                 alpha: float = 1.0, **kw):
        super().__init__(cometric=BLRSoftAbs(X_train, VAR, alpha),
                         l=l, N_fx=N_fx, gamma=gamma, N_run=N_run, **kw)

    def U(self, z: Tensor) -> Tensor:
        return U(z)


class BLRFHMC(FHMC):
    def __init__(self, cometric, l, N_fx, gamma, N_run, **kw):
        super().__init__(_UNUSED_TARGET, cometric, l, N_fx, gamma, N_run, **kw)

    def U(self, z: Tensor) -> Tensor:
        return U(z)


class BLRFHMCInitial(FHMC_initial):
    def __init__(self, cometric, l, N_fx, gamma, N_run, **kw):
        super().__init__(_UNUSED_TARGET, cometric, l, N_fx, gamma, N_run, **kw)

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
        return BLRHMC(mass=params["mass"], l=params["l"], gamma=params["gamma"],
                      N_run=N_run)
    if method == "RHMC":
        return BLRRHMC(l=params["l"], N_fx=params.get("N_fx", 6), gamma=params["gamma"],
                       N_run=N_run, alpha=params.get("alpha", 1.0))
    if method == "FHMC":
        cometric = BLRDualRanders(features=X_train, labels=y_train, var=VAR,
                                  alpha=params.get("alpha", 1.0), beta=params["beta"])
        return BLRFHMC(cometric, l=params["l"], N_fx=params.get("N_fx", 6),
                       gamma=params["gamma"], N_run=N_run, reg=params.get("reg", 0.05),
                       method=params.get("method", "picard"),
                       reduced_flip=params.get("reduced_flip", True))
    if method == "FHMC_INITIAL":
        cometric = BLRDualRanders(features=X_train, labels=y_train, var=VAR,
                                  alpha=params.get("alpha", 1.0), beta=params["beta"])
        return BLRFHMCInitial(cometric, l=params["l"], N_fx=params.get("N_fx", 6),
                              gamma=params["gamma"], N_run=N_run,
                              reduced_flip=params.get("reduced_flip", True))
    raise ValueError(f"unknown method {method!r}")
