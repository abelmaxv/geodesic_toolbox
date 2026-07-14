"""
chi-feng style illustration of (Euclidean) Hamiltonian Monte Carlo on Neal's
2D funnel.

Reuses the generic animation machinery from `hmc_illustration.py` (run_chain +
make_animation) and the experiment's `FunnelHMC`
(internship_experiments/funnel_benchmark2.py).  The 2D funnel is the dim=1 case,
state z = (v, theta), with potential

        U(v, theta) = v^2/18 + v/2 + exp(-v)/2 * theta^2 .

As in the Rosenbrock HMC, the mass matrix is constant (g = I/mass), so the
velocity ellipse is a fixed isotropic circle -- contrast with the RHMC/FHMC
funnel scripts where it adapts to the funnel geometry.

Run:
    python animations/funnel_hmc_illustration.py
Output:
    animations/funnel_hmc_illustration.mp4
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

from hmc_illustration import run_chain, make_animation  # generic machinery  # noqa: E402
from funnel_benchmark2 import FunnelHMC  # noqa: E402


if __name__ == "__main__":
    # ── HMC sampler parameters (edit freely) ─────────────────────────────────
    # Defaults match the tuned operating point from funnel_benchmark2.py.
    DIM = 1          # 2D funnel: state (v, theta)
    MASS = 1         # mass-matrix scale (constant metric g = I / MASS)
    L = 30           # leapfrog steps
    GAMMA = 0.05     # step size
    STD_0 = 1.0      # momentum scale
    BOUNDS = 1e3     # reject proposals with ||z|| beyond this

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 1
    N_ITER = 30
    Z0 = torch.tensor([[1.0, 0.0]])
    XLIM, YLIM = (-6.0, 5.0), (-7.0, 7.0)

    # ── animation timing ─────────────────────────────────────────────────────
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    sampler = FunnelHMC(
        dim=DIM, mass=MASS, l=L, gamma=GAMMA,
        N_run=1, bounds=BOUNDS, std_0=STD_0,
    )

    records = run_chain(sampler, Z0, n_iter=N_ITER, seed=SEED)

    out_path = os.path.join(_HERE, "funnel_hmc_illustration.mp4")
    make_animation(
        records, sampler.U, l=L, gamma=GAMMA,
        xlim=XLIM, ylim=YLIM,
        out_path=out_path,
        title="Hamiltonian Monte Carlo — Neal's funnel (FunnelHMC)",
        reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS,
        xlabel="$v$", ylabel=r"$\theta$",
    )
