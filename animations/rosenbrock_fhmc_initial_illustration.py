"""
chi-feng style illustration of the initial-dynamics Finsler HMC (``FHMC_initial``)
on the Rosenbrock target.

``FHMC_initial`` draws the Randers-biased momentum exactly and integrates the
canonical geodesic Hamiltonian H = U + log sigma_BH + 1/2 F*^2 with the implicit
leapfrog, accepting on the exact H_tilde = H + tau. The initial-velocity
distribution is the Randers density exp(-1/2 (sqrt(v^T G v) + v^T omega)^2), whose
level lines are the decentered ellipses drawn by ``make_animation``.

NOTE: the experiment's tuned point (gamma=0.57) is tuned for sampling efficiency,
not accurate trajectories -- at that step the implicit leapfrog on the stiff
Randers/SoftAbs Hamiltonian diverges. We integrate the same dynamics far more
accurately here (small gamma, many fixed-point iterations); ``run_chain_fhmc``
also drops any residual divergent arc (see DIVERGENCE_BOUND).

Run:
    python animations/rosenbrock_fhmc_initial_illustration.py
Output:
    animations/rosenbrock_fhmc_initial_illustration.mp4
"""

from __future__ import annotations

import os
import sys

import torch
from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from animation_utils import make_animation, run_chain_fhmc  # noqa: E402
from geodesic_toolbox import FHMC_initial  # noqa: E402
from geodesic_toolbox.cometric import RosenbrockDualRanders  # noqa: E402


def U_rosenbrock(z: Tensor) -> Tensor:
    x, y = z[:, 0], z[:, 1]
    return (100 * (y - x**2) ** 2 + (1 - x) ** 2) / 20


def target_rosenbrock(z: Tensor) -> Tensor:
    return torch.exp(-U_rosenbrock(z))


if __name__ == "__main__":
    # ── FHMC_initial sampler parameters (edit freely) ────────────────────────
    L = 60                # leapfrog steps
    GAMMA = 0.05          # step size
    N_FX = 30             # implicit fixed-point iterations
    BETA = 0.4            # Randers drift strength (Finsler asymmetry)
    ALPHA = 1.0           # SoftAbs base-metric parameter
    REDUCED_FLIP = True   # reduced momentum flip on rejection

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 0
    N_ITER = 22
    Z0 = torch.tensor([[-2.0, 4.0]])
    XLIM, YLIM = (-3.5, 3.5), (-1.0, 9.0)

    # ── animation timing ─────────────────────────────────────────────────────
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    cometric = RosenbrockDualRanders(alpha=ALPHA, beta=BETA)
    sampler = FHMC_initial(
        target_rosenbrock, cometric, l=L, N_fx=N_FX, gamma=GAMMA, N_run=1,
        reduced_flip=REDUCED_FLIP,
    )

    torch.manual_seed(SEED)
    records = run_chain_fhmc(sampler, Z0, n_iter=N_ITER, seed=SEED)

    out_path = os.path.join(_HERE, "rosenbrock_fhmc_initial_illustration.mp4")
    make_animation(
        records, U_rosenbrock, l=L, gamma=GAMMA, beta=BETA,
        xlim=XLIM, ylim=YLIM,
        out_path=out_path,
        title="Initial-dynamics Finsler HMC — Rosenbrock (FHMC_initial)",
        reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS,
    )
