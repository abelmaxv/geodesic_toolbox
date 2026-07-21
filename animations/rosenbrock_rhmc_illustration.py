"""
chi-feng style illustration of Riemannian Hamiltonian Monte Carlo (RHMC) on the
Rosenbrock target. The momentum is drawn from N(0, g(z)) with the SoftAbs metric
and integrated with the implicit (Riemannian) leapfrog; the velocity ellipse is
the true covariance of v ~ N(0, g(z)^{-1}).

NOTE: the experiment's tuned point (gamma=0.57, N_fx=6) is tuned for sampling
efficiency, not accurate trajectories -- at that step the implicit leapfrog on
the stiff SoftAbs metric diverges. We integrate the same dynamics far more
accurately here (small gamma, more fixed-point iterations) for a faithful,
gliding-along-the-valley illustration.

Run:
    python animations/rosenbrock_rhmc_illustration.py
Output:
    animations/rosenbrock_rhmc_illustration.mp4
"""

from __future__ import annotations

import os
import sys

import torch
from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from animation_utils import make_animation_hmc, run_chain_rhmc  # noqa: E402
from geodesic_toolbox import ImplicitRHMCSampler  # noqa: E402
from geodesic_toolbox.cometric import RosenbrockSoftAbs  # noqa: E402


class RosenbrockRHMC(ImplicitRHMCSampler):
    """RHMC on the Rosenbrock target with the SoftAbs metric."""

    def __init__(self, l: int, N_fx: int, gamma: float, N_run: int,
                 alpha: float = 1.0, bounds: float = 1e3, std_0: float = 1.,
                 beta_0: float = 1., pbar: bool = False, skip_acceptance: bool = False):
        super().__init__(
            cometric=RosenbrockSoftAbs(alpha=alpha),
            l=l, N_fx=N_fx, gamma=gamma, N_run=N_run,
            bounds=bounds, std_0=std_0, beta_0=beta_0,
            pbar=pbar, skip_acceptance=skip_acceptance,
        )

    def U(self, z: Tensor) -> Tensor:
        x, y = z[:, 0], z[:, 1]
        return (100 * (y - x**2) ** 2 + (1 - x) ** 2) / 20


if __name__ == "__main__":
    # ── RHMC sampler parameters (edit freely) ────────────────────────────────
    L = 60           # leapfrog steps
    GAMMA = 0.10     # step size
    N_FX = 25        # implicit fixed-point iterations
    ALPHA = 1.0      # SoftAbs base-metric parameter
    STD_0 = 1.0      # momentum scale
    BOUNDS = 1e3     # reject proposals with ||z|| beyond this

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 2
    N_ITER = 22
    Z0 = torch.tensor([[-2.0, 4.0]])
    XLIM, YLIM = (-3.5, 3.5), (-1.0, 9.0)

    # ── ellipse / animation timing ───────────────────────────────────────────
    VEL_SCALE = 1.0  # 1.0 = true metric scale for the velocity ellipse + arrow
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    sampler = RosenbrockRHMC(
        l=L, N_fx=N_FX, gamma=GAMMA, N_run=1, alpha=ALPHA, bounds=BOUNDS, std_0=STD_0,
    )

    records = run_chain_rhmc(sampler, Z0, n_iter=N_ITER, seed=SEED)

    out_path = os.path.join(_HERE, "rosenbrock_rhmc_illustration.mp4")
    make_animation_hmc(
        records, sampler.U, l=L, gamma=GAMMA,
        xlim=XLIM, ylim=YLIM,
        out_path=out_path,
        title="Riemannian HMC — Rosenbrock target (RosenbrockRHMC)",
        reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS, vel_scale=VEL_SCALE,
    )
