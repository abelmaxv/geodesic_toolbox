import warnings
import json
import os
import time
from datetime import datetime

import pandas as pd
import torch
from torch import Tensor
import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from botorch.exceptions import OptimizationWarning
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from geodesic_toolbox import ImplicitFHMCUnbiased, ImplicitRHMCSampler, BLRDualRanders, BLRSoftAbs

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)
warnings.filterwarnings("ignore", message=".*not contained to the unit cube.*")
warnings.filterwarnings("ignore", message=".*torch.float32.*")

BURN_IN = 100


# ── Data loading ──────────────────────────────────────────────────────────────

def load_framingham(data_path: str = "../data/framingham/framingham.csv"):
    df = pd.read_csv(data_path).dropna()

    cols_to_drop = [
        "RANDID", "PERIOD",
        "DEATH", "ANGINA", "HOSPMI", "MI_FCHD", "STROKE", "CVD", "HYPERTEN",
        "TIME", "TIMECHD", "TIMECVD", "TIMEDTH", "TIMEHYP",
        "TIMEAP", "TIMEMI", "TIMEMIFC", "TIMESTRK",
    ]

    X = df.drop(columns=cols_to_drop + ["ANYCHD"])
    y = df["ANYCHD"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    X_train = torch.tensor(X_tr, dtype=torch.float64)
    X_test  = torch.tensor(X_te, dtype=torch.float64)
    y_train = torch.tensor(y_tr.values, dtype=torch.float64)
    y_test  = torch.tensor(y_te.values, dtype=torch.float64)

    # Append bias column
    X_train = torch.cat([X_train, torch.ones(X_train.shape[0], 1)], dim=1)
    X_test  = torch.cat([X_test,  torch.ones(X_test.shape[0],  1)], dim=1)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_framingham()
STATE_DIM = X_train.shape[1]   # D features + 1 bias


# ── Samplers ──────────────────────────────────────────────────────────────────

class BLRFHMCUnbiased(ImplicitFHMCUnbiased):
    def __init__(self, features: Tensor, labels: Tensor,
                 l: int, N_fx: int, gamma: float, N_run: int,
                 bounds: float = 1e3, std_0: float = 1., beta_0: float = 1.,
                 pbar: bool = False, skip_acceptance: bool = False,
                 reduced_flip: bool = True,
                 var: float = 1., alpha: float = 1., beta: float = 1.):
        self.features = features
        self.labels   = labels
        self.var      = var
        randers_cometric = BLRDualRanders(
            features=features, labels=labels, var=var, alpha=alpha, beta=beta
        )
        super().__init__(
            randers_cometric=randers_cometric,
            l=l, N_fx=N_fx, gamma=gamma, N_run=N_run,
            bounds=bounds, std_0=std_0, beta_0=beta_0,
            pbar=pbar, skip_acceptance=skip_acceptance,
            reduced_flip=reduced_flip,
        )

    def U(self, beta: Tensor) -> Tensor:
        lh_term    = beta @ self.features.T @ self.labels \
                     - torch.sum(torch.log(1 + torch.exp(beta @ self.features.T)), dim=1)
        prior_term = -0.5 / self.var * (beta ** 2).sum(dim=1)
        return -(lh_term + prior_term)

    def K(self, p: Tensor, z: Tensor) -> Tensor:
        d   = z.shape[1]
        eps = self.randers_cometric.epsilon
        _, w_star, G_star = self.randers_cometric._shared(z)

        v_norm    = torch.einsum("bi,bij,bj->b", p, G_star, p).sqrt()
        F_star    = v_norm + torch.einsum("bi,bi->b", w_star, p)
        F_star_sq = F_star ** 2 + eps ** 2

        L             = torch.linalg.cholesky(G_star)
        y             = torch.linalg.solve_triangular(L, w_star.unsqueeze(-1), upper=False).squeeze(-1)
        alpha         = torch.einsum("bi,bi->b", y, y)
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

        G_inv, w_star, G_star = self.randers_cometric._shared(z)

        L             = torch.linalg.cholesky(G_star)
        y             = torch.linalg.solve_triangular(L, w_star.unsqueeze(-1), upper=False).squeeze(-1)
        b_sq          = torch.einsum("bi,bi->b", y, y)
        alpha_s       = 1.0 - b_sq
        logdet_G_star = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)
        log_sigma_BH  = -0.5 * logdet_G_star

        p_Ginv_p = torch.einsum("bi,bij,bj->b", p, G_inv, p)
        wstar_p  = torch.einsum("bi,bi->b", w_star, p)

        riem_norm          = (wstar_p**2 + p_Ginv_p / alpha_s).sqrt()
        F_star_sq          = (riem_norm + wstar_p)**2 + eps**2
        log_randers_factor = torch.log1p(wstar_p / riem_norm)

        return (
            self.U(z)
            + 0.5  * F_star_sq
            - (d + 1) * log_randers_factor
            + log_sigma_BH
            + 0.5 * d * self.log2pi
        )


class BLRRHMCImplicit(ImplicitRHMCSampler):
    def __init__(self, features: Tensor, labels: Tensor,
                 l: int, N_fx: int, gamma: float, N_run: int,
                 var: float = 1., alpha: float = 1.,
                 bounds: float = 1e3, std_0: float = 1., beta_0: float = 1.,
                 pbar: bool = False, skip_acceptance: bool = False):
        self.features = features
        self.labels   = labels
        self.var      = var
        cometric = BLRSoftAbs(features, var, alpha)
        super().__init__(
            cometric=cometric, l=l, N_fx=N_fx, gamma=gamma, N_run=N_run,
            bounds=bounds, std_0=std_0, beta_0=beta_0,
            pbar=pbar, skip_acceptance=skip_acceptance,
        )

    def U(self, beta: Tensor) -> Tensor:
        lh_term    = beta @ self.features.T @ self.labels \
                     - torch.sum(torch.log(1 + torch.exp(beta @ self.features.T)), dim=1)
        prior_term = -0.5 / self.var * (beta ** 2).sum(dim=1)
        return -(lh_term + prior_term)


# ── Diagnostics ───────────────────────────────────────────────────────────────

def acf(chain: Tensor) -> Tensor:
    N = chain.shape[0]
    norm_chain = chain - chain.mean(dim=0)
    f     = torch.fft.rfft(norm_chain, n=2 * N, dim=0)
    power = f.real ** 2 + f.imag ** 2
    return torch.fft.irfft(power, n=2 * N, dim=0)[:N] / N


def ess(chain: Tensor) -> Tensor:
    if chain.dim() == 3:
        return torch.stack([ess(chain[b]) for b in range(chain.shape[0])])
    N = chain.shape[0]
    gamma_vals = acf(chain)
    var   = gamma_vals[0]
    stuck = var.abs() < 1e-10
    rho   = gamma_vals / var.abs().clamp(min=1e-10).unsqueeze(0)
    positive  = (rho[1:] > 0).cumprod(dim=0)
    ess_vals  = N / (1 + 2 * (rho[1:] * positive).sum(dim=0))
    return torch.where(stuck, torch.ones_like(ess_vals), ess_vals)


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


def compute_accuracy(traj: Tensor, burn_in: int = BURN_IN) -> tuple[float, float]:
    """Predict on X_test using the posterior mean of each chain after burn-in."""
    samples   = traj[:, burn_in + 1:, :]            # (B, N-burn_in, D+1)
    beta_mean = samples.mean(dim=1)                  # (B, D+1)
    logits    = beta_mean @ X_test.T                 # (B, N_test)
    preds     = torch.sigmoid(logits) > 0.5          # (B, N_test)
    acc_per_chain = (preds == y_test.bool()).float().mean(dim=1)   # (B,)
    return acc_per_chain.mean().item(), acc_per_chain.std().item()


# ── Objective and target function ─────────────────────────────────────────────

def min_ess_objective(traj: Tensor, parameters: Tensor) -> Tensor:
    return ess(traj).min(dim=-1).values.mean()


def target_function(
    parameters      : Tensor,
    sampler_factory : callable,
    objective_fn    : callable,
    state_dim       : int,
    N_batch         : int = 10,
    N_run           : int = 10,
) -> Tensor:
    parameters = parameters.squeeze()
    sampler = sampler_factory(parameters, N_run)
    z_0  = torch.randn(N_batch, state_dim)
    traj = sampler.sample(z_0, return_traj=True, progress=False)
    if torch.isnan(traj).any():
        return torch.tensor([0.0])
    return objective_fn(traj, parameters).unsqueeze(0)


# ── Sampler factories ─────────────────────────────────────────────────────────

def make_fhmc_unbiased_factory(reduced_flip: bool = True):
    def factory(parameters: Tensor, N_run: int):
        p = parameters.squeeze()
        return BLRFHMCUnbiased(
            features=X_train, labels=y_train,
            l=round(p[0].item()), N_fx=6,
            gamma=10 ** p[1].item(), N_run=N_run,
            beta=p[2].item(), reduced_flip=reduced_flip,
        )
    return factory


def make_rhmc_factory():
    def factory(parameters: Tensor, N_run: int):
        p = parameters.squeeze()
        return BLRRHMCImplicit(
            features=X_train, labels=y_train,
            l=round(p[0].item()), N_fx=6,
            gamma=10 ** p[1].item(), N_run=N_run,
        )
    return factory


# ── Algorithm configs ─────────────────────────────────────────────────────────

RHMC_CONFIG = {
    "sampler_factory" : make_rhmc_factory(),
    "objective_fn"    : min_ess_objective,
    "bounds"          : torch.tensor([[1., -3.], [20., -0.3]]),
    "param_names"     : ["l", "log10_gamma"],
    "discrete_dims"   : [0],
    "state_dim"       : STATE_DIM,
}

FHMC_REDUCED = {
    "sampler_factory" : make_fhmc_unbiased_factory(reduced_flip=True),
    "objective_fn"    : min_ess_objective,
    "bounds"          : torch.tensor([[1., -3., 0.], [20., -0.3, 1.]]),
    "param_names"     : ["l", "log10_gamma", "beta"],
    "discrete_dims"   : [0],
    "state_dim"       : STATE_DIM,
}

FHMC_NO_FLIP = {
    "sampler_factory" : make_fhmc_unbiased_factory(reduced_flip=False),
    "objective_fn"    : min_ess_objective,
    "bounds"          : torch.tensor([[1., -3., 0.], [20., -0.3, 1.]]),
    "param_names"     : ["l", "log10_gamma", "beta"],
    "discrete_dims"   : [0],
    "state_dim"       : STATE_DIM,
}

ALGORITHMS = {
    "RHMC"         : RHMC_CONFIG,
    "FHMC_NO_FLIP" : FHMC_NO_FLIP,
    "FHMC_REDUCED" : FHMC_REDUCED,
}


# ── Bayesian Optimization ─────────────────────────────────────────────────────

def _params_postfix(parameters: Tensor, param_names: list, discrete_dims: list) -> dict:
    parameters = parameters.flatten()
    names = param_names or [f"p{i}" for i in range(parameters.shape[0])]
    postfix = {}
    for i, name in enumerate(names):
        val = parameters[i].item()
        if name.startswith("log10_"):
            postfix[name[len("log10_"):]] = f"{10 ** val:.4f}"
        elif discrete_dims and i in discrete_dims:
            postfix[name] = round(val)
        else:
            postfix[name] = f"{val:.3f}"
    return postfix


def generate_initial_data(
    n               : int,
    bounds          : Tensor,
    sampler_factory : callable,
    objective_fn    : callable,
    state_dim       : int,
    param_names     : list = None,
    discrete_dims   : list = None,
    N_batch         : int = 10,
    N_run           : int = 10,
):
    d = bounds.shape[1]
    train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, d)
    if discrete_dims:
        for dim in discrete_dims:
            train_X[:, dim] = train_X[:, dim].round()

    results = []
    pbar = tqdm.tqdm(range(n), desc="Initial data")
    for i in pbar:
        postfix = _params_postfix(train_X[i], param_names, discrete_dims)
        pbar.set_postfix(**postfix)
        results.append(target_function(
            train_X[i], sampler_factory, objective_fn,
            state_dim=state_dim, N_batch=N_batch, N_run=N_run,
        ))
    train_Y = torch.stack(results)

    return train_X, train_Y, train_Y.max().item()


def gen_next_points(init_x: Tensor, init_y: Tensor, best_y: Tensor, bounds: Tensor, n_points: int = 1) -> Tensor:
    d = bounds.shape[-1]
    single_model = SingleTaskGP(
        init_x,
        init_y,
        input_transform=Normalize(d=d, bounds=bounds)
    )
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)
    EI = qLogExpectedImprovement(model=single_model, best_f=best_y)
    candidates, _ = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=n_points,
        num_restarts=200,
        raw_samples=512
    )
    return candidates


def run_bayesian_optimization(
    bounds          : Tensor,
    sampler_factory : callable,
    objective_fn    : callable,
    state_dim       : int,
    param_names     : list = None,
    discrete_dims   : list = None,
    n_init          : int = 10,
    n_opt           : int = 10,
    N_batch         : int = 10,
    N_run           : int = 10,
) -> tuple:
    train_x, train_y, best_y = generate_initial_data(
        n_init, bounds, sampler_factory, objective_fn,
        state_dim=state_dim, param_names=param_names, discrete_dims=discrete_dims,
        N_batch=N_batch, N_run=N_run,
    )

    pbar = tqdm.tqdm(range(n_opt), desc="BO iterations")
    for i in pbar:
        candidates = gen_next_points(train_x, train_y, best_y, bounds)
        postfix = _params_postfix(candidates.squeeze(), param_names, discrete_dims)
        pbar.set_postfix(**postfix, best=f"{best_y:.4f}")
        new_y = target_function(
            candidates, sampler_factory, objective_fn,
            state_dim=state_dim, N_batch=N_batch, N_run=N_run,
        )
        train_x = torch.cat([train_x, candidates])
        train_y = torch.cat([train_y, new_y.unsqueeze(0)])
        best_y = train_y.max().item()
        param_str = ", ".join(f"{k}={v}" for k, v in postfix.items())
        tqdm.tqdm.write(f"  iter {i+1:2d} | {param_str} | best={best_y:.4f}")

    best_idx = train_y.squeeze().argmax()
    return train_x[best_idx], best_y


# ── Chain diagnostics ─────────────────────────────────────────────────────────

def compute_diagnostics(
    traj            : Tensor,
    sampler,
    elapsed         : float,
    acceptance_rate : float,
    flip_rate       : float = None,
) -> dict:
    B, _, d = traj.shape
    samples = traj[:, 1:]
    N_run, l = sampler.N_run, sampler.l

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

    acc_mean, acc_std = compute_accuracy(traj)

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
        "test_accuracy"      : acc_mean,
        "test_accuracy_std"  : acc_std,
    }
    if flip_rate is not None:
        diag["flip_rate"] = flip_rate
    return diag


def run_diagnostics(best_x: Tensor, cfg: dict, N_batch: int = 20, N_run: int = 500) -> dict:
    sampler  = cfg["sampler_factory"](best_x, N_run)
    z_0      = torch.randn(N_batch, cfg["state_dim"])
    has_flip = hasattr(sampler, "reduced_flip")

    t0 = time.perf_counter()
    if has_flip:
        traj, acc_rate, flip_rate = sampler.sample(
            z_0, return_traj=True, progress=True, return_acceptance=True, return_flip=True
        )
    else:
        traj, acc_rate = sampler.sample(
            z_0, return_traj=True, progress=True, return_acceptance=True
        )
        flip_rate = None
    elapsed = time.perf_counter() - t0

    return compute_diagnostics(traj, sampler, elapsed, acc_rate, flip_rate)


# ── ACF plot ──────────────────────────────────────────────────────────────────

def run_acf_plots(optimal_params: dict, acf_kwargs: dict, max_lag: int = 400):
    colors     = {"RHMC": "C0", "FHMC_NO_FLIP": "C1", "FHMC_REDUCED": "C2"}
    dim_labels = [r"$\beta_0$", r"$\beta_5$", r"$\beta_{10}$"]
    dim_names  = ["beta0", "beta5", "beta10"]
    dim_indices = [0, 5, 10]

    model_name   = "logistic_regression"
    acf_plot_dir = os.path.join("results", model_name, "acf_plot")
    os.makedirs(acf_plot_dir, exist_ok=True)

    acf_curves_per_algo = {}
    for name, cfg in ALGORITHMS.items():
        best_x  = torch.tensor([optimal_params[name]["params"][k] for k in cfg["param_names"]])
        sampler = cfg["sampler_factory"](best_x, acf_kwargs["N_run"])
        z_0     = torch.randn(acf_kwargs["N_batch"], cfg["state_dim"])

        tqdm.tqdm.write(f"Sampling {name} for ACF plot...")
        traj    = sampler.sample(z_0, return_traj=True, progress=True)
        samples = traj[:, 1:]
        B, N, _ = samples.shape
        lag     = min(max_lag, N)

        acf_curves = []
        for b in range(B):
            g   = acf(samples[b])[:lag]
            rho = g / g[0].abs().clamp(min=1e-10)
            acf_curves.append(rho)
        acf_curves_per_algo[name] = torch.stack(acf_curves).detach()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for dim_idx, (dim_label, dim_name, state_idx) in enumerate(zip(dim_labels, dim_names, dim_indices)):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, acf_curves in acf_curves_per_algo.items():
            lag_count = acf_curves.shape[1]
            lags      = torch.arange(lag_count).numpy()
            c         = colors[name]
            vals      = acf_curves[:, :, state_idx]
            mean_acf  = vals.mean(dim=0).numpy()
            std_acf   = vals.std(dim=0).numpy()
            ax.plot(lags, mean_acf, label=name, color=c, lw=1.5)
            ax.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf, alpha=0.2, color=c)

        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.grid(True, ls="--", color="grey", alpha=0.5)
        ax.set_title(f"ACF — Bayesian logistic regression, dimension {dim_label}")
        ax.legend(fontsize=9)
        plt.tight_layout()

        path = os.path.join(acf_plot_dir, f"logistic_regression_acf_plot_{dim_name}_{timestamp}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"ACF plot saved to {path}")
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    BO_KWARGS   = dict(n_init=20, n_opt=50, N_batch=20, N_run=1000)
    DIAG_KWARGS = dict(N_batch=50, N_run=2000)
    ACF_KWARGS  = dict(N_batch=50, N_run=2000)

    optimal_params  = {}
    all_diagnostics = {}

    for name, cfg in ALGORITHMS.items():
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")
        best_x, best_y = run_bayesian_optimization(**cfg, **BO_KWARGS)
        optimal_params[name] = {
            "params" : {n: best_x[i].item() for i, n in enumerate(cfg["param_names"])},
            "min_ess": best_y,
        }
        print(f"  Running diagnostics...")
        all_diagnostics[name] = run_diagnostics(best_x, cfg, **DIAG_KWARGS)

    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    for name, res in optimal_params.items():
        def _fmt(k, v):
            if k == "l":               return f"l={round(v)}"
            if k.startswith("log10_"): return f"{k[len('log10_'):]}={10**v:.4f}"
            return f"{k}={v:.3f}"
        params_str = ", ".join(_fmt(k, v) for k, v in res["params"].items())

        d = all_diagnostics[name]
        flip_str = f" | flip={d['flip_rate']:.3f}" if "flip_rate" in d else ""
        print(f"{name:15s} | minESS={res['min_ess']:.2f} | {params_str}")
        print(f"{'':15s}   ESS/step={d['ess_per_step']:.4f} (var={d['ess_per_step_var']:.2e})"
              f" | ESS/s={d['ess_per_second']:.1f} (var={d['ess_per_second_var']:.2e})"
              f" | EBFMI={d['ebfmi']:.3f} (var={d['ebfmi_var']:.2e})")
        print(f"{'':15s}   R-hat={d['rhat_max']:.4f}"
              f" | MCSE={d['mcse_max']:.4f} (var={d['mcse_max_var']:.2e})"
              f" | acc={d['acceptance_rate']:.3f}{flip_str}")
        print(f"{'':15s}   test_accuracy={d['test_accuracy']:.4f} ± {d['test_accuracy_std']:.4f}")

    # Save results
    model_name = "logistic_regression"
    logs_dir   = os.path.join("results", model_name, "benchmark_logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log = {}
    for name, res in optimal_params.items():
        d = all_diagnostics[name]
        log[name] = {
            "params"             : res["params"],
            "min_ess"            : res["min_ess"],
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
            "test_accuracy"      : d["test_accuracy"],
            "test_accuracy_std"  : d["test_accuracy_std"],
        }

    log_filename = os.path.join(logs_dir, f"{model_name}_benchmark_log_{timestamp}.json")
    with open(log_filename, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nBenchmark log saved to {log_filename}")

    run_acf_plots(optimal_params, ACF_KWARGS)


if __name__ == "__main__":
    main()
