"""
Shared statistics and machinery for the sampler benchmarks.

Each ``{target}_{method}_benchmark.py`` builds a sampler for one fixed
configuration, runs it, and reports diagnostics through this module:

* ESS / autocorrelation : ``acf``, ``ess_from_chain``, ``ess``,
  ``min_ess_per_step`` (Geyer initial-positive-sequence IAT).
* Convergence           : ``gelman_rubin`` (split-chain R-hat).
* Sample quality        : ``ksd`` (IMQ kernel Stein discrepancy; needs a target
  score) and ``mmd_per_chain`` (per-coordinate RBF MMD; needs reference samples).
* ``funnel_exact_samples`` : exact funnel draws, used as the MMD reference.
* ``NUTS``              : hamiltorch NUTS wrapped in the geodesic .sample() API.
* ``compute_diagnostics`` / ``run_benchmark`` : bundle a run into a dict and a
  JSON log; KSD is added only when a ``score_fn`` is given, MMD only when
  ``reference_samples`` are given.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import time
from datetime import datetime

import torch
from torch import Tensor

MAX_LAG = 150  # autocorrelation truncation window for the ESS estimator


# ── ESS / autocorrelation ─────────────────────────────────────────────────────

def acf(chain: Tensor, max_lag: int = MAX_LAG) -> Tensor:
    """Normalized autocorrelation at lags 0..max_lag for a 1D chain (max_lag is
    capped at len(chain) - 1)."""
    max_lag = min(max_lag, chain.shape[0] - 1)
    chain = chain - chain.mean()
    var = chain.var(unbiased=False)
    return torch.stack([
        (chain[:chain.shape[0] - k] * chain[k:]).mean() / var if k > 0 else torch.ones_like(var)
        for k in range(max_lag + 1)
    ])


def ess_from_chain(chain: Tensor, max_lag: int = MAX_LAG) -> float:
    """ESS = N / IAT, IAT = 1 + 2 sum_{k>=1} ACF(k), summed over Geyer's initial
    positive sequence (truncated at the first negative lag). A constant chain
    (every proposal rejected) carries no information, so ESS = 0."""
    if chain.var(unbiased=False) <= 0:
        return 0.0
    ac = acf(chain, max_lag)
    neg = ac[1:] < 0
    first_neg = int(neg.long().argmax().item()) if bool(neg.any()) else 0
    cutoff = first_neg if first_neg > 0 else max_lag
    iat = 1.0 + 2.0 * ac[1:cutoff + 1].sum().item()
    return chain.shape[0] / max(iat, 1.0)


def ess(traj: Tensor, max_lag: int = MAX_LAG) -> Tensor:
    """Per-chain, per-coordinate ESS. traj is (N, d) or (B, N, d)."""
    if traj.dim() == 3:
        return torch.stack([ess(traj[b], max_lag) for b in range(traj.shape[0])])
    d = traj.shape[1]
    return torch.tensor([ess_from_chain(traj[:, k], max_lag) for k in range(d)])


def gelman_rubin(traj: Tensor) -> Tensor:
    """Split-chain potential scale reduction factor R-hat, per coordinate."""
    B, N, d = traj.shape
    if B < 2:
        return torch.ones(d)
    chain_means = traj.mean(dim=1)
    grand_mean = chain_means.mean(dim=0)
    W = (traj - chain_means.unsqueeze(1)).pow(2).sum(dim=(0, 1)) / (B * (N - 1))
    B_hat = N / (B - 1) * (chain_means - grand_mean).pow(2).sum(dim=0)
    var_hat = (N - 1) / N * W + B_hat / N
    return (var_hat / W.clamp(min=1e-10)).sqrt()


def min_ess_per_step(chain: Tensor, l: int, N_run: int) -> Tensor:
    return ess(chain).min() / (N_run * l)


# ── KSD (IMQ kernel Stein discrepancy, Gorham & Mackey 2017) ───────────────────

def imq_stein_kernel(X: Tensor, Y: Tensor, score_X: Tensor, score_Y: Tensor,
                     c: float = 1.0, beta: float = -0.5) -> Tensor:
    """Stein kernel matrix k_0(x_i, y_j) for the IMQ base kernel
    k(x, y) = (c^2 + ||x - y||^2)^beta. X (n, d), Y (m, d), scores same shape."""
    d = X.shape[1]
    diff = X.unsqueeze(1) - Y.unsqueeze(0)
    sq = (diff ** 2).sum(-1)
    u = c ** 2 + sq
    u_b1 = u ** (beta - 1)
    k = u ** beta
    grad_x_k = 2 * beta * u_b1.unsqueeze(-1) * diff          # grad_y k = -grad_x k
    trace_term = -2 * beta * (d * u_b1 + 2 * (beta - 1) * u ** (beta - 2) * sq)
    sx = score_X.unsqueeze(1)
    sy = score_Y.unsqueeze(0)
    return (
        trace_term
        + (sx * (-grad_x_k)).sum(-1)
        + (sy * grad_x_k).sum(-1)
        + (sx * sy).sum(-1) * k
    )


def ksd(X: Tensor, score_fn, c: float = 1.0, beta: float = -0.5,
        max_points: int | None = None, generator: torch.Generator | None = None) -> float:
    """IMQ kernel Stein discrepancy of X w.r.t. the target whose score is
    score_fn ((n, d) -> (n, d)). V-statistic; returns KSD (not KSD^2).
    KSD is O(n^2 d), so subsample to max_points when set."""
    if max_points is not None and X.shape[0] > max_points:
        idx = torch.randperm(X.shape[0], generator=generator)[:max_points]
        X = X[idx]
    s = score_fn(X)
    K0 = imq_stein_kernel(X, X, s, s, c=c, beta=beta)
    return float(K0.mean().clamp_min(0).sqrt())


# ── MMD (per-coordinate RBF, median-heuristic bandwidth) ───────────────────────

def mmd2_per_coord(X: Tensor, Y: Tensor) -> Tensor:
    """Per-coordinate RBF MMD^2, vectorized over d."""
    n, d = X.shape
    dxx = X.unsqueeze(1) - X.unsqueeze(0)
    dyy = Y.unsqueeze(1) - Y.unsqueeze(0)
    dxy = X.unsqueeze(1) - Y.unsqueeze(0)
    pooled = torch.cat([dxx.abs().reshape(-1, d), dxy.abs().reshape(-1, d)], dim=0)
    sigma = pooled.median(dim=0).values.clamp(min=1e-6)
    inv = 1.0 / (2 * sigma ** 2)
    Kxx = torch.exp(-dxx ** 2 * inv)
    Kyy = torch.exp(-dyy ** 2 * inv)
    Kxy = torch.exp(-dxy ** 2 * inv)
    return (Kxx.mean(dim=(0, 1)) + Kyy.mean(dim=(0, 1)) - 2 * Kxy.mean(dim=(0, 1))).clamp(min=0)


def mmd_per_chain(traj_samples: Tensor, reference: Tensor, n_sub: int = 400,
                  generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """Per chain: mean per-coordinate MMD^2 and the coord-0 MMD^2 (the funnel
    neck v-marginal). Returns (mmd_mean (B,), mmd_coord0 (B,))."""
    B = traj_samples.shape[0]
    mmd_mean = torch.empty(B)
    mmd_0 = torch.empty(B)
    for b in range(B):
        s = traj_samples[b]
        if s.shape[0] > n_sub:
            s = s[torch.randperm(s.shape[0], generator=generator)[:n_sub]]
        per_coord = mmd2_per_coord(s, reference)
        mmd_mean[b] = per_coord.mean()
        mmd_0[b] = per_coord[0]
    return mmd_mean, mmd_0


def funnel_exact_samples(n: int, dim: int, seed: int | None = None) -> Tensor:
    """Exact funnel draws: v ~ N(0, 9), theta_i | v ~ N(0, e^v). Shape (n, dim+1)."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    v = torch.randn(n, generator=g) * 3.0
    theta = torch.randn(n, dim, generator=g) * torch.exp(v / 2).unsqueeze(1)
    return torch.cat([v.unsqueeze(1), theta], dim=1)


# ── NUTS adapter (hamiltorch, geodesic .sample() interface) ────────────────────

class NUTS:
    """hamiltorch NUTS exposing the geodesic samplers' .sample() signature.
    Trajectory length is adaptive, so l = 1 makes 'ESS/step' read as ESS/sample.

    Parameters
    ----------
    log_prob : callable (d,) -> scalar
        Unnormalized log density of a single state.
    """

    def __init__(self, log_prob, N_run: int, burn: int = 200, init_step: float = 0.05,
                 max_steps: int = 1024, accept: float = 0.8):
        self.log_prob = log_prob
        self.N_run = N_run
        self.burn = burn
        self.init_step = init_step
        self.max_steps = max_steps
        self.accept = accept
        self.l = 1

    def sample(self, z_0: Tensor, return_traj: bool = False, progress: bool = False,
               return_acceptance: bool = False):
        import hamiltorch
        chains, accs = [], []
        for b in range(z_0.shape[0]):
            hamiltorch.set_random_seed(b)
            with contextlib.redirect_stdout(io.StringIO()):
                samples, acc = hamiltorch.sample(
                    log_prob_func=self.log_prob, params_init=z_0[b].clone(),
                    num_samples=self.N_run + self.burn, burn=self.burn,
                    step_size=self.init_step, num_steps_per_sample=self.max_steps,
                    sampler=hamiltorch.Sampler.HMC_NUTS,
                    desired_accept_rate=self.accept,
                    store_on_GPU=False, debug=2, verbose=False,
                )
            chains.append(torch.stack(samples)[:self.N_run])
            accs.append(float(acc))
        traj = torch.cat([z_0.unsqueeze(1), torch.stack(chains)], dim=1)
        acc_rate = sum(accs) / len(accs)
        if return_traj:
            return (traj, acc_rate) if return_acceptance else traj
        return (traj[:, -1], acc_rate) if return_acceptance else traj[:, -1]


# ── Diagnostics ────────────────────────────────────────────────────────────────

def compute_diagnostics(traj: Tensor, sampler, elapsed: float, acceptance_rate: float,
                        flip_rate: float | None = None, score_fn=None,
                        reference_samples: Tensor | None = None, metric_seed: int = 12345,
                        ksd_n_sub: int = 400, ksd_n_pooled: int = 1500,
                        mmd_n_sub: int = 400) -> dict:
    """Bundle a run into a diagnostics dict. ESS / R-hat / acceptance are always
    computed; KSD is added when ``score_fn`` is given, MMD when
    ``reference_samples`` are given (per-coordinate, coord 0 reported separately)."""
    samples = traj[:, 1:]                        # drop the initial state
    B, N_run, d = samples.shape
    l = sampler.l

    ess_vals = ess(samples)
    per_chain_min_ess = ess_vals.min(dim=-1).values
    n_trapped = int((per_chain_min_ess == 0).sum().item())
    rhat = gelman_rubin(samples)

    def _mv(x):
        return x.mean().item(), x.var().item()

    def _q(x):
        q = torch.quantile(x, torch.tensor([0.10, 0.50, 0.90]))
        return q[1].item(), q[0].item(), q[2].item()

    min_ess, min_ess_var = _mv(per_chain_min_ess)
    min_ess_median, min_ess_q10, min_ess_q90 = _q(per_chain_min_ess)

    diag = {
        "acceptance_rate": acceptance_rate,
        "elapsed_s": elapsed,
        "n_chains": B,
        "n_run": N_run,
        "min_ess": min_ess,
        "min_ess_var": min_ess_var,
        "min_ess_median": min_ess_median,
        "min_ess_q10": min_ess_q10,
        "min_ess_q90": min_ess_q90,
        "ess_per_step": (per_chain_min_ess / (N_run * l)).mean().item(),
        "ess_per_second": (per_chain_min_ess / elapsed).mean().item(),
        "rhat_max": rhat.max().item(),
        "rhat_coord0": rhat[0].item(),
        "n_trapped": n_trapped,
    }
    if flip_rate is not None:
        diag["flip_rate"] = flip_rate

    if score_fn is not None:
        g = torch.Generator().manual_seed(metric_seed)
        per_chain_ksd = torch.tensor(
            [ksd(samples[b], score_fn, max_points=ksd_n_sub, generator=g) for b in range(B)])
        pooled_ksd = ksd(samples.reshape(-1, d), score_fn, max_points=ksd_n_pooled, generator=g)
        ksd_median, ksd_q10, ksd_q90 = _q(per_chain_ksd)
        diag.update(ksd_median=ksd_median, ksd_q10=ksd_q10, ksd_q90=ksd_q90,
                    ksd_pooled=pooled_ksd)

    if reference_samples is not None:
        g = torch.Generator().manual_seed(metric_seed)
        mmd_mean, mmd_0 = mmd_per_chain(samples, reference_samples, n_sub=mmd_n_sub, generator=g)
        diag.update(mmd_median=mmd_mean.median().item(), mmd_coord0=mmd_0.mean().item())

    return diag


def run_benchmark(name: str, sampler, z_0: Tensor, params: dict, *, target: str,
                  score_fn=None, reference_samples: Tensor | None = None,
                  log_dir: str = "results", print_summary: bool = True) -> dict:
    """Run one sampler configuration, compute diagnostics, print a summary and
    write a JSON log to ``{log_dir}/{target}_{method}/``. ``name`` labels the
    method (e.g. ``"RHMC"``)."""
    has_flip = hasattr(sampler, "reduced_flip")
    t0 = time.perf_counter()
    if has_flip:
        traj, acc, flip = sampler.sample(z_0, return_traj=True, progress=False,
                                         return_acceptance=True, return_flip=True)
    else:
        traj, acc = sampler.sample(z_0, return_traj=True, progress=False,
                                   return_acceptance=True)
        flip = None
    elapsed = time.perf_counter() - t0

    diag = compute_diagnostics(traj, sampler, elapsed, acc, flip,
                               score_fn=score_fn, reference_samples=reference_samples)

    if print_summary:
        print_diagnostics(name, target, params, diag)

    out_dir = os.path.join(log_dir, f"{target}_{name.lower()}")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {"target": target, "method": name, "params": params, "diagnostics": diag}
    path = os.path.join(out_dir, f"{target}_{name.lower()}_{ts}.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"log saved to {path}")
    return diag


def print_diagnostics(name: str, target: str, params: dict, d: dict) -> None:
    flip = f" | flip={d['flip_rate']:.3f}" if "flip_rate" in d else ""
    trap = f" | trapped={d['n_trapped']}/{d['n_chains']}" if d["n_trapped"] else ""
    print("=" * 78)
    print(f"  {target} / {name}   {params}")
    print("=" * 78)
    print(f"  minESS med={d['min_ess_median']:8.1f} "
          f"[{d['min_ess_q10']:.1f},{d['min_ess_q90']:.1f}] | "
          f"ESS/s={d['ess_per_second']:8.3f} | ESS/step={d['ess_per_step']:.4f}")
    print(f"  Rhat_max={d['rhat_max']:.4f} (coord0: {d['rhat_coord0']:.4f}) | "
          f"acc={d['acceptance_rate']:.3f}{flip}{trap}")
    if "ksd_median" in d:
        print(f"  KSD med={d['ksd_median']:.4f} "
              f"[{d['ksd_q10']:.4f},{d['ksd_q90']:.4f}] pooled={d['ksd_pooled']:.4f}")
    if "mmd_median" in d:
        print(f"  MMD med={d['mmd_median']:.2e}  MMD_coord0={d['mmd_coord0']:.2e}")
