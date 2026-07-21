"""
Illustration of FHMC (corrected non-canonical dynamics) on the rotational Randers
bump ring. The target is the Busemann-Hausdorff density sigma_BH(z) =
det G*(z)^{-1/2} (via slogdet, so off-ring states are rejected per-sample), which
reproduces the ring density. For the corrected dynamics the sampled (cyan) and
launch (green) velocity arrows coincide.

Two versions are produced, differing only in the flip scheme on rejection:

    ring_fhmc_noflipreduction_illustration.mp4   reduced_flip = False
    ring_fhmc_flipreduction_illustration.mp4     reduced_flip = True

Run:    python animations/ring_fhmc_illustration.py
"""

from __future__ import annotations

import os
import sys

import torch
from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from animation_utils import make_animation, make_ring_U, run_chain_fhmc  # noqa: E402
from geodesic_toolbox import FHMC  # noqa: E402
from geodesic_toolbox.cometric import DualRandersMetrics, RandersBumpRotational  # noqa: E402


if __name__ == "__main__":
    # ── metric / sampler params ──────────────────────────────────────────────
    BETA = 0.9         # Randers rotational drift strength
    SCALING = 0.5
    L = 3             # implicit midpoint steps per proposal
    N_FX = 10           # fixed-point (picard) iterations per step
    GAMMA = 0.005   # step size
    REG = 0.05         # smoothing of the p = 0 singularity of the field

    # ── chain / view ─────────────────────────────────────────────────────────
    # Both variants share the seed for a controlled comparison; seed 3 separates
    # them early and shows the contrast (fewer flips / more accepts with reduction).
    SEED = 3
    N_ITER = 34
    Z0 = torch.tensor([[0.0, 1.0]])
    XLIM = YLIM = (-1.5, 1.5)

    # ── momentum ellipse / animation timing ──────────────────────────────────
    VEL_SCALE = 0.025  # ring velocities are large (|v|~13); shrink for on-frame ellipses
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    metric = RandersBumpRotational(beta=BETA, scale=SCALING)
    cometric = DualRandersMetrics(metric)
    U_bg = make_ring_U(metric)

    def bh_target(z: Tensor) -> Tensor:
        """Busemann-Hausdorff density sigma_BH(z) = det G*(z)^{-1/2}."""
        _, logabsdet = torch.linalg.slogdet(cometric.G_star(z))
        return torch.exp(-0.5 * logabsdet)

    for reduced_flip, tag, label in [
        (False, "noflipreduction", "without flip reduction"),
        (True, "flipreduction", "with flip reduction"),
    ]:
        sampler = FHMC(
            bh_target, cometric, l=L, N_fx=N_FX, gamma=GAMMA, N_run=1,
            pbar=False, reduced_flip=reduced_flip, reg=REG,
        )
        records = run_chain_fhmc(sampler, Z0, n_iter=N_ITER, seed=SEED)

        out_path = os.path.join(_HERE, f"ring_fhmc_{tag}_illustration.mp4")
        make_animation(
            records, U_bg, l=L, gamma=GAMMA, beta=BETA,
            xlim=XLIM, ylim=YLIM,
            out_path=out_path,
            title=f"Corrected-dynamics FHMC on the Randers bump ring — {label}",
            reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS, vel_scale=VEL_SCALE,
        )
