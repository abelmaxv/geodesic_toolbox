"""
chi-feng style illustration of Euclidean Hamiltonian Monte Carlo (HMC) on the
Rosenbrock target. The momentum is drawn from N(0, g(z)) with the constant mass
g = I / MASS (an isotropic circle), integrated with the standard leapfrog.

Run:
    python animations/rosenbrock_hmc_illustration.py
Output:
    animations/rosenbrock_hmc_illustration.mp4
"""

from __future__ import annotations

import os
import sys

import torch
from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from animation_utils import make_animation_hmc, run_chain_hmc  # noqa: E402
from geodesic_toolbox import HMCSampler  # noqa: E402
from geodesic_toolbox.cometric import IdentityCoMetric  # noqa: E402


class RosenbrockHMC(HMCSampler):
    """HMC on the Rosenbrock target with a constant mass metric g = I / mass."""

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
        return (100 * (y - x**2) ** 2 + (1 - x) ** 2) / 20


if __name__ == "__main__":
    # ── HMC sampler parameters (edit freely) ─────────────────────────────────
    MASS = 20        # mass-matrix scale (constant metric g = I / MASS)
    L = 60           # leapfrog steps
    GAMMA = 0.05     # step size
    STD_0 = 1.0      # momentum scale
    BOUNDS = 1e3     # reject proposals with ||z|| beyond this

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 1
    N_ITER = 30
    Z0 = torch.tensor([[-1.5, 2.5]])
    XLIM, YLIM = (-2.5, 2.5), (-0.7, 5.0)

    # ── animation timing ─────────────────────────────────────────────────────
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    sampler = RosenbrockHMC(
        mass=MASS, l=L, gamma=GAMMA, N_run=1, bounds=BOUNDS, std_0=STD_0,
    )

    records = run_chain_hmc(sampler, Z0, n_iter=N_ITER, seed=SEED)

    out_path = os.path.join(_HERE, "rosenbrock_hmc_illustration.mp4")
    make_animation_hmc(
        records, sampler.U, l=L, gamma=GAMMA,
        xlim=XLIM, ylim=YLIM,
        out_path=out_path,
        title="Hamiltonian Monte Carlo — Rosenbrock target (RosenbrockHMC)",
        reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS, vel_scale=L * GAMMA,
    )
