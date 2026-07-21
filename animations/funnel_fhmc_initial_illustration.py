"""
chi-feng style illustration of the initial-dynamics Finsler HMC (``FHMC_initial``)
on Neal's 2D funnel (the dim=1 case, state z = (v, theta)):

        U(v, theta) = v^2/18 + v/2 + exp(-v)/2 * theta^2 .

``FHMC_initial`` draws the Randers-biased momentum exactly and integrates the
canonical geodesic Hamiltonian with the implicit leapfrog, accepting on the exact
H_tilde. The initial-velocity distribution is the Randers density
exp(-1/2 (sqrt(v^T G v) + v^T omega)^2); its decentered ellipses adapt to the
funnel geometry.

NOTE on alpha: the experiment's SoftAbs alpha=1e6 is too stiff for the implicit
leapfrog (it diverges), so we soften it (alpha~10) for a stable illustration;
``run_chain_fhmc`` still guards any residual divergence.

Run:
    python animations/funnel_fhmc_initial_illustration.py
Output:
    animations/funnel_fhmc_initial_illustration.mp4
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
from geodesic_toolbox.cometric import FunnelDualRanders  # noqa: E402

DIM = 1  # 2D funnel: state (v, theta)


def U_funnel(z: Tensor) -> Tensor:
    v = z[:, 0]
    theta = z[:, 1:]
    return v**2 / 18 + (DIM / 2) * v + torch.exp(-v) / 2 * (theta**2).sum(dim=1)


def target_funnel(z: Tensor) -> Tensor:
    return torch.exp(-U_funnel(z))


if __name__ == "__main__":
    # ── FHMC_initial sampler parameters (edit freely) ────────────────────────
    L = 25           # leapfrog steps
    GAMMA = 0.05     # step size
    N_FX = 10        # implicit fixed-point iterations
    BETA = 0.2       # Randers drift strength (Finsler asymmetry)
    ALPHA = 10.0     # SoftAbs sharpness (softened from the experiment's 1e6)
    REDUCED_FLIP = True  # reduced momentum flip on rejection

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 0
    N_ITER = 24
    Z0 = torch.tensor([[1.0, 0.0]])
    XLIM, YLIM = (-6.0, 5.0), (-7.0, 7.0)

    # ── ellipse / animation timing ───────────────────────────────────────────
    VEL_SCALE = 1.0  # 1.0 = true metric scale for the Randers density level lines
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    cometric = FunnelDualRanders(dim=DIM, alpha=ALPHA, beta=BETA)
    sampler = FHMC_initial(
        target_funnel, cometric, l=L, N_fx=N_FX, gamma=GAMMA, N_run=1,
        reduced_flip=REDUCED_FLIP,
    )

    torch.manual_seed(SEED)
    records = run_chain_fhmc(sampler, Z0, n_iter=N_ITER, seed=SEED)

    out_path = os.path.join(_HERE, "funnel_fhmc_initial_illustration.mp4")
    make_animation(
        records, U_funnel, l=L, gamma=GAMMA, beta=BETA,
        xlim=XLIM, ylim=YLIM,
        out_path=out_path,
        title="Initial-dynamics Finsler HMC — Neal's funnel (FHMC_initial)",
        reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS, vel_scale=VEL_SCALE,
        xlabel="$v$", ylabel=r"$\theta$",
    )
