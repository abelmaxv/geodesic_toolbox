"""
Illustration of FHMC (corrected non-canonical dynamics) on Neal's 2D funnel
(dim = 1, state z = (v, theta)). FHMC takes an explicit target exp(-U_funnel) and
adds the Busemann-Hausdorff volume term internally. For the corrected dynamics
the field's position block is the inverse Legendre transform, so the sampled
(cyan) and launch (green) velocity arrows coincide.

NOTE on alpha: the experiment's SoftAbs alpha = 1e6 is too stiff for the implicit
integrator, so we soften it (alpha ~ 10).

Run:    python animations/funnel_fhmc_illustration.py
Output: animations/funnel_fhmc_illustration.mp4  (falls back to .gif)
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
from geodesic_toolbox.cometric import FunnelDualRanders  # noqa: E402

DIM = 1  # 2D funnel: state (v, theta)


def U_funnel(z: Tensor) -> Tensor:
    v = z[:, 0]
    theta = z[:, 1:]
    return v**2 / 18 + (DIM / 2) * v + torch.exp(-v) / 2 * (theta**2).sum(dim=1)


def target_funnel(z: Tensor) -> Tensor:
    return torch.exp(-U_funnel(z))


if __name__ == "__main__":
    # ── FHMC sampler parameters (edit freely) ────────────────────────────────
    # Tuned for BETA = 0.9, where the field is ~10x stiffer than at 0.2. The
    # integrator is exact (|log alpha| ~ 1e-5); acceptance is limited only by arcs
    # that leave the domain, which shorter T (= gamma*l) avoids.
    L = 20                # implicit midpoint steps per proposal (T = gamma*l = 0.06)
    GAMMA = 0.01         # step size (beta=0.9 field is ~10x stiffer than at 0.2)
    N_FX =  10             # Newton iterations per step
    METHOD = "newton"     # fixed-point solver (newton handles the stiff field)
    BETA = 0.9            # Randers drift strength (Finsler asymmetry)
    ALPHA = 10.0          # SoftAbs sharpness (softened from the experiment's 1e6)
    REG = 10**-5            # smoothing of the p = 0 singularity of the field
    REDUCED_FLIP = True  # reduced momentum flip on rejection

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 2
    N_ITER = 24
    Z0 = torch.tensor([[1.0, 0.0]])
    XLIM, YLIM = (-6.0, 5.0), (-7.0, 7.0)

    # ── animation timing ─────────────────────────────────────────────────────
    VEL_SCALE = 1.0       # 1.0 = true metric scale for the Randers density ellipses
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    cometric = FunnelDualRanders(dim=DIM, alpha=ALPHA, beta=BETA)
    sampler = FHMC(
        target_funnel, cometric, l=L, N_fx=N_FX, gamma=GAMMA, N_run=1,
        pbar=False, reduced_flip=REDUCED_FLIP, method=METHOD, reg=REG,
    )

    # run_chain_fhmc no longer seeds internally, so seed here for
    # reproducibility.
    #torch.manual_seed(SEED)
    records = run_chain_fhmc(sampler, Z0, n_iter=N_ITER, seed=SEED)

    out_path = os.path.join(_HERE, "funnel_fhmc_illustration.mp4")
    make_animation(
        records, U_funnel, l=L, gamma=GAMMA, beta=BETA,
        xlim=XLIM, ylim=YLIM,
        out_path=out_path,
        title=None,
        reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS, vel_scale=VEL_SCALE,
        xlabel="$v$", ylabel=r"$\theta$",
    )
