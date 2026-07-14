"""
chi-feng style illustration of *Riemannian* Hamiltonian Monte Carlo on Neal's
2D funnel.

Reuses the generic animation machinery from `rhmc_illustration.py` (run_chain +
make_animation) and the experiment's `FunnelRHMC`
(internship_experiments/funnel_benchmark2.py).  The 2D funnel is the dim=1 case,
state z = (v, theta), with potential

        U(v, theta) = v^2/18 + v/2 + exp(-v)/2 * theta^2 .

The velocity ellipse is the true covariance ellipse of N(0, g(z)^{-1}); on the
funnel the SoftAbs metric makes it tight in the neck (low v) and wide in the
mouth (high v), so you can watch it breathe with the local geometry.

NOTE on alpha: the experiment uses alpha=1e6 in the SoftAbs metric (~ exact
|Hessian|), which is so stiff that the implicit leapfrog diverges and most
trajectories blow up. For a faithful *illustration* we soften it (alpha~10): the
metric is still strongly position-dependent and funnel-adapted, but smooth enough
to integrate without blow-ups (run_chain also guards any residual divergence).

Run:
    python animations/funnel_rhmc_illustration.py
Output:
    animations/funnel_rhmc_illustration.mp4
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.join(os.path.dirname(_HERE), "internship_experiments")
for _p in (_HERE, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rhmc_illustration import run_chain, make_animation  # generic machinery  # noqa: E402
from funnel_benchmark2 import FunnelRHMC  # noqa: E402


if __name__ == "__main__":
    # ── RHMC sampler parameters (edit freely) ────────────────────────────────
    DIM = 1          # 2D funnel: state (v, theta)
    L = 40           # leapfrog steps
    GAMMA = 0.05     # step size
    N_FX = 15        # implicit fixed-point iterations
    ALPHA = 10.0     # SoftAbs sharpness (softened from the experiment's 1e6)
    STD_0 = 1.0      # momentum scale
    BOUNDS = 1e3     # reject proposals with ||z|| beyond this

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 2
    N_ITER = 24
    Z0 = torch.tensor([[1.0, 0.0]])
    XLIM, YLIM = (-6.0, 5.0), (-7.0, 7.0)

    # ── ellipse / animation timing ───────────────────────────────────────────
    VEL_SCALE = 1.0  # 1.0 = true metric scale for the velocity ellipse + arrow
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    sampler = FunnelRHMC(
        dim=DIM, l=L, N_fx=N_FX, gamma=GAMMA, N_run=1,
        alpha=ALPHA, bounds=BOUNDS, std_0=STD_0,
    )

    records = run_chain(sampler, Z0, n_iter=N_ITER, seed=SEED)

    out_path = os.path.join(_HERE, "funnel_rhmc_illustration.mp4")
    make_animation(
        records, sampler.U, l=L, gamma=GAMMA,
        xlim=XLIM, ylim=YLIM,
        out_path=out_path,
        title="Riemannian HMC — Neal's funnel (FunnelRHMC)",
        reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS, vel_scale=VEL_SCALE,
        xlabel="$v$", ylabel=r"$\theta$",
    )
