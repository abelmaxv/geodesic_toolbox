"""Worker module for the beta-sweep in rosenbrock_benchmark3.ipynb.

The sweep is parallelised with ProcessPoolExecutor. On macOS the default
``spawn`` start method re-imports the worker function *by reference*, so it must
live in an importable module (functions defined inside the notebook live in
``__main__`` and cannot be pickled). Keep this file in sync with the notebook's
Step-3 definitions; the notebook imports ``eval_beta`` from here.
"""

import torch
from torch import Tensor

from geodesic_toolbox import ImplicitFHMCUnbiased, RosenbrockDualRanders


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
        return (100 * (y - x**2)**2 + (1 - x)**2) / 20

    def K(self, p: Tensor, z: Tensor) -> Tensor:
        d   = z.shape[1]
        eps = self.randers_cometric.epsilon
        _, w_star, G_star = self.randers_cometric._shared(z)   # ONE eigh

        v_norm = torch.einsum("bi,bij,bj->b", p, G_star, p).sqrt()
        F_star = v_norm + torch.einsum("bi,bi->b", w_star, p)
        F_star_sq = F_star ** 2 + eps ** 2

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

        L             = torch.linalg.cholesky(G_star)
        y             = torch.linalg.solve_triangular(
                            L, w_star.unsqueeze(-1), upper=False).squeeze(-1)
        b_sq          = torch.einsum("bi,bi->b", y, y)   # |ω*|²_{G*⁻¹} = 1 – α_s
        alpha_s       = 1.0 - b_sq
        logdet_G_star = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)

        log_sigma_BH  = -0.5 * logdet_G_star

        p_Ginv_p = torch.einsum("bi,bij,bj->b", p, G_inv, p)   # p^T G^{-1} p
        wstar_p  = torch.einsum("bi,bi->b", w_star, p)          # ω* · p

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


# ── ESS estimation ──────────────────────────────────────────────────────────

def acf(chain: Tensor) -> Tensor:
    N = chain.shape[0]
    norm_chain = chain - chain.mean(dim=0)
    f = torch.fft.rfft(norm_chain, n=2 * N, dim=0)
    power = f.real ** 2 + f.imag ** 2
    return torch.fft.irfft(power, n=2 * N, dim=0)[:N] / N


def ess(chain: Tensor) -> Tensor:
    if chain.dim() == 3:
        return torch.stack([ess(chain[b]) for b in range(chain.shape[0])])
    N = chain.shape[0]
    gamma_vals = acf(chain)
    var = gamma_vals[0]
    stuck = var.abs() < 1e-10
    rho = gamma_vals / var.abs().clamp(min=1e-10).unsqueeze(0)
    positive = (rho[1:] > 0).cumprod(dim=0)
    ess_vals = N / (1 + 2 * (rho[1:] * positive).sum(dim=0))
    n_moves = (chain[1:] != chain[:-1]).double().sum(dim=0)
    ess_vals = torch.minimum(ess_vals, n_moves + 1.0)
    return torch.where(stuck, torch.ones_like(ess_vals), ess_vals)


def per_chain_ess(traj: Tensor, burnin_frac: float = 0.15) -> Tensor:
    """Per-chain, per-coordinate ESS, after discarding burn-in.

    Returns a tensor of shape (batch_size, d). Keeping every coordinate (rather
    than collapsing to a min) is what lets us see the directional effect of the
    Randers wind: on the Rosenbrock the curved x-direction is the bottleneck the
    method targets, and min-over-coordinates hides its improvement.
    """
    if burnin_frac > 0:
        burn = int(burnin_frac * traj.shape[1])
        traj = traj[:, burn:, :]
    return ess(traj)


def per_chain_min_ess(traj: Tensor, burnin_frac: float = 0.15) -> Tensor:
    """Per-chain min-over-dimensions ESS, after discarding burn-in (kept for
    backward compatibility; prefer per_chain_ess to retain the per-coordinate view)."""
    return per_chain_ess(traj, burnin_frac).min(dim=-1).values


def eval_beta_traj(beta: float, gamma: float, l: int, reduced_flip: bool,
                    N_run: int = 1000, batch_size: int = 50,
                    z_0: Tensor | None = None, num_threads: int = 1,
                    seed: int | None = None):
    """Run one FHMC chain-batch at a given beta and return the raw trajectory.

    Unlike `eval_beta`, this applies no ESS estimator or burn-in itself --
    it hands back the trajectory so the caller can score it with whatever
    estimator/z_0 it needs (e.g. to stay comparable with a Riemannian HMC
    baseline computed elsewhere with a different estimator).

    seed : int | None
        Each beta runs in its own process with an independent, OS-derived RNG
        state by default; without a fixed seed, beta=0 (which should be
        numerically equivalent to RHMC -- see rosenbrock_benchmark3.ipynb's
        sanity check) only matches an RHMC baseline up to ordinary MC noise.
        Pass the same seed used for the RHMC baseline to get a like-for-like
        comparison instead of two independent stochastic realizations.
    """
    torch.set_num_threads(num_threads)
    if seed is not None:
        torch.manual_seed(seed)
    sampler = RosenbrockFHMCUnbiased(
        l=l, N_fx=6, gamma=gamma, N_run=N_run, beta=beta, reduced_flip=reduced_flip
    )
    if z_0 is None:
        z_0 = torch.zeros(batch_size, 2)
    traj, acc, flip = sampler.sample(
        z_0, return_traj=True, return_acceptance=True, return_flip=True
    )
    return traj, acc, flip


def eval_beta(beta: float, gamma: float, l: int, reduced_flip: bool,
              N_run: int = 1000, batch_size: int = 50, burnin_frac: float = 0.15,
              num_threads: int = 1):
    """Run one FHMC chain-batch at a given beta and return diagnostics.

    Parameters
    ----------
    num_threads : int
        Intra-process torch thread count. With ProcessPoolExecutor each worker
        should stay single-threaded (=1) so the N processes do not oversubscribe
        the cores by each spawning a full BLAS thread pool.

    Returns
    -------
    ess_per_chain : Tensor (batch_size, d)
        Per-chain, per-coordinate ESS (e.g. columns = [ESS_x, ESS_y]).
    acc : float
    flip : float
    """
    torch.set_num_threads(num_threads)
    sampler = RosenbrockFHMCUnbiased(
        l=l, N_fx=6, gamma=gamma, N_run=N_run, beta=beta, reduced_flip=reduced_flip
    )
    z_0 = torch.zeros(batch_size, 2)
    traj, acc, flip = sampler.sample(
        z_0, return_traj=True, return_acceptance=True, return_flip=True
    )
    ess_per_chain = per_chain_ess(traj, burnin_frac=burnin_frac)
    return ess_per_chain, acc, flip
