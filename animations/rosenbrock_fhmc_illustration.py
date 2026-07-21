"""
Illustration of FHMC (corrected non-canonical dynamics) on the Rosenbrock
target. FHMC takes an explicit target exp(-U_rosenbrock) and adds the
Busemann-Hausdorff volume term internally. For the corrected dynamics the field's
position block is the inverse Legendre transform, so the sampled (cyan) and
launch (green) velocity arrows coincide.

Run:    python animations/rosenbrock_fhmc_illustration.py
Output: animations/rosenbrock_fhmc_illustration.mp4  (falls back to .gif)
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
from geodesic_toolbox import FHMC  # noqa: E402
from geodesic_toolbox.cometric import RosenbrockDualRanders  # noqa: E402


def U_rosenbrock(z: Tensor) -> Tensor:
    x, y = z[:, 0], z[:, 1]
    return (100 * (y - x**2) ** 2 + (1 - x) ** 2) / 20


def target_rosenbrock(z: Tensor) -> Tensor:
    return torch.exp(-U_rosenbrock(z))


if __name__ == "__main__":
    # ── FHMC sampler parameters (edit freely) ───────────────────────
    # The corrected field is stiffer near p = 0 and each implicit-midpoint step
    # evaluates a full state Jacobian, so we keep a modest step / step count.
    L = 30                # implicit midpoint steps per proposal
    GAMMA = 0.03          # step size
    N_FX = 12             # fixed-point (picard) iterations per step
    BETA = 0.4            # Randers drift strength (Finsler asymmetry)
    ALPHA = 1.0           # SoftAbs base-metric parameter
    REG = 0.05            # smoothing of the p = 0 singularity of the field
    REDUCED_FLIP = True   # reduced momentum flip on rejection

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 0
    N_ITER = 22
    Z0 = torch.tensor([[-2.0, 4.0]])
    XLIM, YLIM = (-3.5, 3.5), (-1.0, 9.0)

    # ── animation timing ─────────────────────────────────────────────────────
    VEL_SCALE = 1.0       # 1.0 = true metric scale for the Randers density ellipses
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    cometric = RosenbrockDualRanders(alpha=ALPHA, beta=BETA)
    sampler = FHMC(
        target_rosenbrock, cometric, l=L, N_fx=N_FX, gamma=GAMMA, N_run=1,
        pbar=False, reduced_flip=REDUCED_FLIP, reg=REG,
    )

    # run_chain_fhmc no longer seeds internally, so seed here for
    # reproducibility.
    torch.manual_seed(SEED)
    records = run_chain_fhmc(sampler, Z0, n_iter=N_ITER, seed=SEED)

    out_path = os.path.join(_HERE, "rosenbrock_fhmc_illustration.mp4")
    make_animation(
        records, U_rosenbrock, l=L, gamma=GAMMA, beta=BETA,
        xlim=XLIM, ylim=YLIM,
        out_path=out_path,
        title="Corrected-dynamics FHMC — Rosenbrock (FHMC)",
        reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS, vel_scale=VEL_SCALE,
    )
