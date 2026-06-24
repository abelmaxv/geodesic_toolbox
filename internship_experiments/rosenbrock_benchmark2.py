import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import torch
from torch import Tensor

from geodesic_toolbox import (
    HMCSampler, IdentityCoMetric,
    ImplicitFHMCUnbiased, ImplicitRHMCSampler, RosenbrockDualRanders, RosenbrockSoftAbs,
)

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*torch.float32.*")


# ── Fixed sampler parameters ─────────────────────────────────────────────────
# Edit these to set the operating point of each sampler. Keys must match what
# each ALGORITHMS[*]["sampler_factory"] reads (see the factory classes below).

PARAMS = {
    "HMC": {
        "mass"  : 20,
        "l"     : 25,
        "gamma" : 0.46,
    },
    "RHMC": {
        "l"     : 2,
        "gamma" : 0.57,
    },
    "FHMC": {
        "beta"  : 0.6,
        "l"     : 2,
        "gamma" : 0.57,
    },
    "FHMC_REDUCED": {
        "beta"  : 0.5,
        "l"     : 2,
        "gamma" : 0.57,
    },
}


# ── Samplers ──────────────────────────────────────────────────────────────────

class RosenbrockFHMCUnbiased(ImplicitFHMCUnbiased):
    def __init__(self, l: int, N_fx: int, gamma: float, N_run: int,
                 bounds: float = 1e3, std_0: float = 1., beta_0: float = 1.,
                 pbar: bool = False, skip_acceptance: bool = False,
                 reduced_flip: bool = True, alpha: float = 1., beta: float = 1.):
        randers_cometric = RosenbrockDualRanders(alpha=alpha, beta=beta)
        super().__init__(
            randers_cometric=randers_cometric,
            l=l, N_fx=N_fx, gamma=gamma, N_run=N_run,
            bounds=bounds, std_0=std_0, beta_0=beta_0,
            pbar=pbar, skip_acceptance=skip_acceptance,
            reduced_flip=reduced_flip,
        )

    def U(self, z):
        x, y = z[:, 0], z[:, 1]
        return (100*(y - x**2)**2 + (1 - x)**2) / 20

    def K(self, p: Tensor, z: Tensor) -> Tensor:
        d   = z.shape[1]
        eps = self.randers_cometric.epsilon
        _, w_star, G_star = self.randers_cometric._shared(z)   # ONE eigh

        # F* without re-eigh
        v_norm = torch.einsum("bi,bij,bj->b", p, G_star, p).sqrt()
        F_star = v_norm + torch.einsum("bi,bi->b", w_star, p)
        F_star_sq = F_star ** 2 + eps ** 2     # avoid the throw-away outer sqrt

        # BH terms via one Cholesky of G* (G* is already in hand)
        L = torch.linalg.cholesky(G_star)
        y = torch.linalg.solve_triangular(L, w_star.unsqueeze(-1), upper=False).squeeze(-1)
        alpha = torch.einsum("bi,bi->b", y, y)
        logdet_G_star = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)

        return (0.5 * F_star_sq
                - 0.5 * (d + 1) * torch.log1p(-alpha)
                - 0.5 * logdet_G_star
                + 0.5 * d * self.log2pi)

    def H(self, z: Tensor, p: Tensor) -> Tensor:
        return self.U(z) + self.K(p, z)

    def H_tilde(self, z: Tensor, p: Tensor) -> Tensor:
        d   = z.shape[1]
        eps = self.randers_cometric.epsilon

        G_inv, w_star, G_star = self.randers_cometric._shared(z)   # ONE eigh

        # ── logdet G* and α_s via one Cholesky of G* ──────────────────────────
        L             = torch.linalg.cholesky(G_star)
        y             = torch.linalg.solve_triangular(
                            L, w_star.unsqueeze(-1), upper=False).squeeze(-1)
        b_sq          = torch.einsum("bi,bi->b", y, y)   # |ω*|²_{G*⁻¹} = 1 – α_s
        alpha_s       = 1.0 - b_sq
        logdet_G_star = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)

        # BCS identity: det G · det G* = α_s^{–(d+1)}
        #   ⟹ ½(d+1) log α_s + ½ log det G = –½ log det G*
        log_sigma_BH  = -0.5 * logdet_G_star

        # ── momentum scalars, without materialising G* ────────────────────────
        p_Ginv_p = torch.einsum("bi,bij,bj->b", p, G_inv, p)   # p^T G^{-1} p
        wstar_p  = torch.einsum("bi,bi->b", w_star, p)          # ω* · p

        # p^T G* p = (ω*·p)² + p^T G^{-1} p / α_s   (from the MDL form of G*)
        riem_norm = (wstar_p**2 + p_Ginv_p / alpha_s).sqrt()
        F_star_sq = (riem_norm + wstar_p)**2 + eps**2
        log_randers_factor = torch.log1p(wstar_p / riem_norm)

        return (
            self.U(z)
            + 0.5  * F_star_sq
            - (d + 1) * log_randers_factor
            + log_sigma_BH
            + 0.5 * d * self.log2pi
        )


class RosenbrockRHMC(ImplicitRHMCSampler):
    def __init__(self, l: int, N_fx: int, gamma: float, N_run: int,
                 alpha: float = 1.0, bounds: float = 1e3, std_0: float = 1.,
                 beta_0: float = 1., pbar: bool = False, skip_acceptance: bool = False):
        cometric = RosenbrockSoftAbs(alpha=alpha)
        super().__init__(
            cometric=cometric, l=l, N_fx=N_fx, gamma=gamma, N_run=N_run,
            bounds=bounds, std_0=std_0, beta_0=beta_0,
            pbar=pbar, skip_acceptance=skip_acceptance,
        )

    def U(self, z: Tensor) -> Tensor:
        x, y = z[:, 0], z[:, 1]
        return (100*(y - x**2)**2 + (1 - x)**2) / 20


class RosenbrockHMC(HMCSampler):
    def __init__(self, mass: float, l: int, gamma: float, N_run: int,
                 bounds: float = 1e3, std_0: float = 1., beta_0: float = 1.,
                 pbar: bool = False, skip_acceptance: bool = False):
        super().__init__(
            cometric=IdentityCoMetric(coscale=mass, is_diag=False),
            l=l, gamma=gamma, N_run=N_run,
            bounds=bounds, std_0=std_0, beta_0=beta_0,
            pbar=pbar, skip_acceptance=skip_acceptance,
        )

    def U(self, z: Tensor) -> Tensor:
        x, y = z[:, 0], z[:, 1]
        return (100 * (y - x**2)**2 + (1 - x)**2) / 20

    def leapfrog_trajectory(self, z_0: Tensor) -> Tensor:
        v_0 = self.sample_momentum(z_0)
        traj_q, _ = self.leapfrog(z_0, v_0, return_traj=True)
        return traj_q   # (B, l+1, d)


# ── Diagnostics ───────────────────────────────────────────────────────────────

MAX_LAG = 150  # truncation window for the Geyer estimator below — matches
                # rosenbrock_sampling.ipynb's max_lag.


def acf(chain: Tensor, max_lag: int) -> Tensor:
    """Normalized autocorrelation at lags 0..max_lag for a 1D chain — matches
    rosenbrock_sampling.ipynb's acf()."""
    chain = chain - chain.mean()
    var = chain.var(unbiased=False)
    return torch.stack([
        (chain[:chain.shape[0] - k] * chain[k:]).mean() / var if k > 0 else torch.ones_like(var)
        for k in range(max_lag + 1)
    ])


def ess_from_chain(chain: Tensor, max_lag: int) -> float:
    """
    ESS = N / IAT,  IAT = 1 + 2 * sum_{k>=1} ACF(k).
    Sum truncated at the first negative lag (Geyer's initial positive
    sequence) — matches rosenbrock_sampling.ipynb's ess_from_chain exactly.
    """
    ac = acf(chain, max_lag)
    neg = ac[1:] < 0
    first_neg = int(neg.long().argmax().item()) if bool(neg.any()) else 0
    cutoff = first_neg if first_neg > 0 else max_lag
    iat = 1.0 + 2.0 * ac[1:cutoff + 1].sum().item()
    return chain.shape[0] / max(iat, 1.0)


def ess(traj: Tensor, max_lag: int = MAX_LAG) -> Tensor:
    """Per-chain, per-coordinate ESS via ess_from_chain — batched the same way
    rosenbrock_sampling.ipynb's total_ess/mean_ess loop over chains and dims."""
    if traj.dim() == 3:
        return torch.stack([ess(traj[b], max_lag) for b in range(traj.shape[0])])
    d = traj.shape[1]
    return torch.tensor([ess_from_chain(traj[:, k], max_lag) for k in range(d)])


def gelman_rubin(traj: Tensor) -> Tensor:
    B, N, d = traj.shape
    if B < 2:
        return torch.ones(d)
    chain_means = traj.mean(dim=1)
    grand_mean  = chain_means.mean(dim=0)
    W     = (traj - chain_means.unsqueeze(1)).pow(2).sum(dim=(0, 1)) / (B * (N - 1))
    B_hat = N / (B - 1) * (chain_means - grand_mean).pow(2).sum(dim=0)
    var_hat = (N - 1) / N * W + B_hat / N
    return (var_hat / W.clamp(min=1e-10)).sqrt()


def min_ess_per_step(chain: Tensor, l: int, N_run: int) -> Tensor:
    return ess(chain).min() / (N_run * l)


# ── Sampler factories ─────────────────────────────────────────────────────────
# Plain classes (not closures) so they can be pickled and sent to the worker
# processes that parallelize the per-algorithm diagnostics in main().

class FHMCUnbiasedFactory:
    def __init__(self, reduced_flip: bool = True, N_fx: int = 6):
        self.reduced_flip = reduced_flip
        self.N_fx         = N_fx

    def __call__(self, params: dict, N_run: int):
        return RosenbrockFHMCUnbiased(
            l=params["l"], N_fx=self.N_fx, gamma=params["gamma"], N_run=N_run,
            beta=params["beta"], reduced_flip=self.reduced_flip,
        )


class RHMCFactory:
    def __init__(self, N_fx: int = 6):
        self.N_fx = N_fx

    def __call__(self, params: dict, N_run: int):
        return RosenbrockRHMC(
            l=params["l"], N_fx=self.N_fx, gamma=params["gamma"], N_run=N_run,
        )


class HMCFactory:
    def __call__(self, params: dict, N_run: int):
        return RosenbrockHMC(
            mass=params["mass"], l=params["l"], gamma=params["gamma"], N_run=N_run,
        )


# ── Algorithm configs ─────────────────────────────────────────────────────────

ALGORITHMS = {
    "HMC"          : {"sampler_factory": HMCFactory()},
    "RHMC"         : {"sampler_factory": RHMCFactory()},
    "FHMC"         : {"sampler_factory": FHMCUnbiasedFactory(reduced_flip=False)},
    "FHMC_REDUCED" : {"sampler_factory": FHMCUnbiasedFactory(reduced_flip=True)},
}


# ── Chain diagnostics ─────────────────────────────────────────────────────────

def compute_diagnostics(
    traj            : Tensor,
    sampler,
    elapsed         : float,
    acceptance_rate : float,
    flip_rate       : float = None,
) -> dict:
    B, _, d = traj.shape
    samples = traj[:, 1:]                 # drop z_0
    N_run, l = samples.shape[1], sampler.l

    ess_vals = ess(samples)
    per_chain_min_ess = ess_vals.min(dim=-1).values

    U_vals     = sampler.U(samples.reshape(-1, d)).reshape(B, N_run)
    delta_U_sq = (U_vals[:, 1:] - U_vals[:, :-1]).pow(2)
    per_chain_ebfmi = delta_U_sq.mean(dim=1) / U_vals.var(dim=1).clamp(min=1e-10)

    rhat = gelman_rubin(samples)

    sd_vals = samples.std(dim=1)
    per_chain_mcse = (sd_vals / ess_vals.sqrt()).max(dim=-1).values

    per_chain_ess_per_step   = per_chain_min_ess / (N_run * l)
    per_chain_ess_per_second = per_chain_min_ess / elapsed

    def _mean_var(per_chain: Tensor) -> tuple:
        return per_chain.mean().item(), per_chain.var().item()

    min_ess,        min_ess_var        = _mean_var(per_chain_min_ess)
    ess_per_step,   ess_per_step_var   = _mean_var(per_chain_ess_per_step)
    ess_per_second, ess_per_second_var = _mean_var(per_chain_ess_per_second)
    ebfmi_val,      ebfmi_var          = _mean_var(per_chain_ebfmi)
    mcse_max,       mcse_max_var       = _mean_var(per_chain_mcse)

    diag = {
        "acceptance_rate"    : acceptance_rate,
        "min_ess"            : min_ess,
        "min_ess_var"        : min_ess_var,
        "ess_per_step"       : ess_per_step,
        "ess_per_step_var"   : ess_per_step_var,
        "ess_per_second"     : ess_per_second,
        "ess_per_second_var" : ess_per_second_var,
        "ebfmi"              : ebfmi_val,
        "ebfmi_var"          : ebfmi_var,
        "rhat_max"           : rhat.max().item(),
        "mcse_max"           : mcse_max,
        "mcse_max_var"       : mcse_max_var,
    }
    if flip_rate is not None:
        diag["flip_rate"] = flip_rate
    return diag


def run_diagnostics(params: dict, cfg: dict, N_batch: int = 20, N_run: int = 500) -> dict:
    sampler  = cfg["sampler_factory"](params, N_run)
    z_0      = torch.randn(N_batch, 2)
    has_flip = hasattr(sampler, "reduced_flip")

    t0 = time.perf_counter()
    if has_flip:
        traj, acc_rate, flip_rate = sampler.sample(
            z_0, return_traj=True, progress=False, return_acceptance=True, return_flip=True
        )
    else:
        traj, acc_rate = sampler.sample(
            z_0, return_traj=True, progress=False, return_acceptance=True
        )
        flip_rate = None
    elapsed = time.perf_counter() - t0

    return compute_diagnostics(traj, sampler, elapsed, acc_rate, flip_rate)


def _run_algorithm_diagnostics(name: str, cfg: dict, params: dict, diag_kwargs: dict) -> tuple:
    """Module-level so it can be pickled and run in its own process — the
    diagnostics for each algorithm are independent of one another."""
    return name, run_diagnostics(params, cfg, **diag_kwargs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DIAG_KWARGS = dict(N_batch=100, N_run=1000)

    print(f"{'='*55}")
    print("  FIXED SAMPLER PARAMETERS")
    print(f"{'='*55}")
    for name, params in PARAMS.items():
        print(f"  {name:15s} : {params}")

    # The 4 algorithms are independent (each is internally vectorized over its
    # own batch of chains), so run their diagnostics concurrently.
    print(f"\nRunning diagnostics for {len(ALGORITHMS)} algorithms in parallel...")
    all_diagnostics = {}
    with ProcessPoolExecutor(max_workers=len(ALGORITHMS)) as executor:
        futures = [
            executor.submit(_run_algorithm_diagnostics, name, cfg, PARAMS[name], DIAG_KWARGS)
            for name, cfg in ALGORITHMS.items()
        ]
        for future in as_completed(futures):
            name, diag = future.result()
            all_diagnostics[name] = diag
            print(f"  Done: {name}")

    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    for name, params in PARAMS.items():
        params_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in params.items())

        d = all_diagnostics[name]
        flip_str = f" | flip={d['flip_rate']:.3f}" if "flip_rate" in d else ""
        print(f"{name:15s} | {params_str}")
        print(f"{'':15s}   minESS={d['min_ess']:.2f} (var={d['min_ess_var']:.2e})"
              f" | ESS/step={d['ess_per_step']:.4f} (var={d['ess_per_step_var']:.2e})"
              f" | ESS/s={d['ess_per_second']:.1f} (var={d['ess_per_second_var']:.2e})"
              f" | EBFMI={d['ebfmi']:.3f} (var={d['ebfmi_var']:.2e})")
        print(f"{'':15s}   R-hat={d['rhat_max']:.4f}"
              f" | MCSE={d['mcse_max']:.4f} (var={d['mcse_max_var']:.2e})"
              f" | acc={d['acceptance_rate']:.3f}{flip_str}")

    # Save results
    model_name = "rosenbrock_benchmark2"
    logs_dir   = os.path.join("results", model_name, "benchmark_logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diag_tag  = "dim2_Nbatch{N_batch}_Nrun{N_run}".format(**DIAG_KWARGS)

    log = {"diagnostics_config": DIAG_KWARGS}
    for name, params in PARAMS.items():
        d = all_diagnostics[name]
        log[name] = {
            "params"             : params,
            "min_ess"            : d["min_ess"],
            "min_ess_var"        : d["min_ess_var"],
            "ess_per_step"       : d["ess_per_step"],
            "ess_per_step_var"   : d["ess_per_step_var"],
            "ess_per_second"     : d["ess_per_second"],
            "ess_per_second_var" : d["ess_per_second_var"],
            "ebfmi"              : d["ebfmi"],
            "ebfmi_var"          : d["ebfmi_var"],
            "rhat_max"           : d["rhat_max"],
            "mcse_max"           : d["mcse_max"],
            "mcse_max_var"       : d["mcse_max_var"],
            "acceptance_rate"    : d["acceptance_rate"],
            "flip_rate"          : d.get("flip_rate", None),
        }

    log_filename = os.path.join(logs_dir, f"{model_name}_log_{diag_tag}_{timestamp}.json")
    with open(log_filename, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nBenchmark log saved to {log_filename}")


if __name__ == "__main__":
    main()
