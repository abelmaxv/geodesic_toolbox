"""
chi-feng style illustration of *Finsler* Hamiltonian Monte Carlo (FHMC) on Neal's
2D funnel.

Reuses the generic animation machinery from `fhmc_illustration.py` (run_chain +
make_animation) and the experiment's `FunnelFHMCUnbiased`
(internship_experiments/funnel_benchmark2.py).  The 2D funnel is the dim=1 case,
state z = (v, theta), with potential

        U(v, theta) = v^2/18 + v/2 + exp(-v)/2 * theta^2 .

The initial-velocity distribution is the Randers density
exp(-1/2 (sqrt(v^T G v) + v^T omega)^2); its level lines are decentered ellipses
that adapt to the funnel geometry.

NOTE on alpha: as in the funnel RHMC script, the experiment's SoftAbs alpha=1e6 is
too stiff for the implicit leapfrog (it diverges), so we soften it (alpha~10) for
a stable, drawable illustration; run_chain still guards any residual divergence.

Run:
    python animations/funnel_fhmc_illustration.py
Output:
    animations/funnel_fhmc_illustration.mp4
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

from fhmc_illustration import run_chain, make_animation  # generic machinery  # noqa: E402
from funnel_benchmark2 import FunnelFHMCUnbiased  # noqa: E402


if __name__ == "__main__":
    # ── FHMC sampler parameters (edit freely) ────────────────────────────────
    DIM = 1          # 2D funnel: state (v, theta)
    L = 25           # leapfrog steps
    GAMMA = 0.05     # step size
    N_FX = 10        # implicit fixed-point iterations
    BETA = 0.2      # Randers drift strength (Finsler asymmetry)
    ALPHA = 10.0     # SoftAbs sharpness (softened from the experiment's 1e6)
    STD_0 = 1.0      # momentum scale
    REDUCED_FLIP = True  # reduced momentum flip on rejection

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 0
    N_ITER = 24
    Z0 = torch.tensor([[1.0, 0.0]])
    XLIM, YLIM = (-6.0, 5.0), (-7.0, 7.0)

    # ── ellipse / animation timing ───────────────────────────────────────────
    VEL_SCALE = 1.0  # 1.0 = true metric scale for the Randers density level lines
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    sampler = FunnelFHMCUnbiased(
        dim=DIM, l=L, N_fx=N_FX, gamma=GAMMA, N_run=1,
        beta=BETA, alpha=ALPHA, std_0=STD_0, reduced_flip=REDUCED_FLIP,
    )

    records = run_chain(sampler, Z0, n_iter=N_ITER, seed=SEED)

    out_path = os.path.join(_HERE, "funnel_fhmc_illustration.mp4")
    make_animation(
        records, sampler.U, l=L, gamma=GAMMA, beta=BETA,
        xlim=XLIM, ylim=YLIM,
        out_path=out_path,
        title="Finsler HMC — Neal's funnel (FunnelFHMCUnbiased)",
        reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS, vel_scale=VEL_SCALE,
        xlabel="$v$", ylabel=r"$\theta$",
    )
