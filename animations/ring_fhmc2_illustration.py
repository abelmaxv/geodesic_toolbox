"""
chi-feng style illustration of ImplicitFHMCUnbiased2Reg on the rotational
Randers *bump* ring, in the same single-trajectory style as
fhmc_illustration.py / funnel_fhmc_illustration.py (decentered momentum
ellipses, implicit Finsler leapfrog arc, accept/reject, forward/backward
badge).

ImplicitFHMCUnbiased2Reg's own H_tilde_reg (in samplers.py) has no separate
target potential U(z) -- which is exactly right here, since the ring's density
comes purely from the Finsler (Busemann-Hausdorff) volume form baked into the
metric (see ImplicitFHMCUnbiased.U), so no override is needed in this file
(unlike the Rosenbrock / funnel twins).

Two versions are produced, differing only in the momentum-flip scheme:

    ring_fhmc2_noflipreduction_illustration.mp4   reduced_flip = False
    ring_fhmc2_flipreduction_illustration.mp4      reduced_flip = True

The bump base metric concentrates mass on a ring; the Randers one-form omega adds
a rotational (clockwise) drift, so the initial-velocity density is *decentered*
(the cyan ellipses are shifted off the current point). With reduced_flip = False
the momentum direction reverses on every rejection (watch the FORWARD/BACKWARD
badge toggle); with reduced_flip = True it rarely flips, so the trajectory
circulates coherently around the ring.

Reuses run_chain + make_animation from fhmc_illustration.py. Because the true
Busemann-Hausdorff potential U is not finite off the ring (its Cholesky fails),
the *background* is drawn from a robust ring proxy (the base-metric log-det, as
in FHMCUnbiased_bump_metric.ipynb); the trajectory/arcs themselves come from the
real sampler dynamics.

Run:
    python animations/ring_fhmc2_illustration.py
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
from geodesic_toolbox import ImplicitFHMCUnbiased2Reg  # noqa: E402
from geodesic_toolbox.cometric import DualRandersMetrics, RandersBumpRotational  # noqa: E402


def make_ring_U(metric):
    """Robust ring background: exp(-U) reproduces the base-metric log-det ring.

    The real target U (Busemann-Hausdorff) is undefined off the ring, so for the
    *background image only* we use the base-metric log-det (always finite),
    normalised so exp(-U) lands in (0, 1].
    """
    def U(grid):
        g = metric.base_cometric(grid.float())
        log_det = torch.log(1.0 / (g[:, 0] * g[:, 1] + 1e-6) + 1e-6)
        return log_det.max() - log_det
    return U


if __name__ == "__main__":
    # ── metric / sampler params ──────────────────────────────────────────────
    BETA = 0.9        # Randers rotational drift strength
    SCALING = 0.5
    L = 50             # leapfrog steps per proposal
    N_FX = 6          # implicit fixed-point iterations
    GAMMA = 0.0005     # step size
    DELTA_REL = 0.05   # H_tilde_reg smoothing (relative to sqrt(d))

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 0
    N_ITER = 34
    Z0 = torch.tensor([[0.0, 1.0]])
    XLIM = YLIM = (-1.5, 1.5)

    # ── momentum ellipse / animation timing ──────────────────────────────────
    VEL_SCALE = 0.025  # ring velocities are large (|v|~13); shrink for on-frame ellipses
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    metric = RandersBumpRotational(beta=BETA, scale=SCALING)
    cometric = DualRandersMetrics(metric)
    U_bg = make_ring_U(metric)

    for reduced_flip, tag, label in [
        (False, "noflipreduction", "without flip reduction"),
        (True, "flipreduction", "with flip reduction"),
    ]:
        sampler = ImplicitFHMCUnbiased2Reg(
            cometric, l=L, N_fx=N_FX, gamma=GAMMA, N_run=1,
            pbar=False, reduced_flip=reduced_flip, delta_rel=DELTA_REL,
        )
        # Cyan arrow = sampled Randers (Legendre) velocity dH/dp, a genuine draw
        # from the displayed decentered ellipses (which make_animation reflects,
        # w -> -w, during backward phases so they always show the distribution of
        # the actual initial travel velocity). Green arrow = the leapfrog's own
        # launch velocity dH_tilde/dp(v_half): FHMC2 integrates H_tilde, whose
        # -(d+1)log1p correction rotates the launch away from the sampled
        # velocity (up to ~150 deg here; an exact-H_tilde effect, not the delta
        # smoothing -- the arc itself is N_fx-converged and energy-conserving),
        # so the two arrows genuinely disagree on many iterations.
        records = run_chain(sampler, Z0, n_iter=N_ITER, seed=SEED,
                            arrow_velocity="legendre")

        out_path = os.path.join(_HERE, f"ring_fhmc2_{tag}_illustration.mp4")
        make_animation(
            records, U_bg, l=L, gamma=GAMMA, beta=BETA,
            xlim=XLIM, ylim=YLIM,
            out_path=out_path,
            title=f"Finsler HMC on the Randers bump ring — {label}",
            reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS, vel_scale=VEL_SCALE,
        )
