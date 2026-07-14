"""
chi-feng style illustration of *Finsler* Hamiltonian Monte Carlo on the
Rosenbrock target, using the ``ImplicitFHMCUnbiased2Reg`` variant.

This is the twin of ``fhmc_illustration.py``. The only difference is the
sampler: ``ImplicitFHMCUnbiased2Reg`` drives its implicit leapfrog with a
delta-regularised *corrected* Randers Hamiltonian ``H_tilde_reg`` (a smoothed
"K_tilde" kinetic term) instead of the bare geodesic Hamiltonian ``H`` ("K").
Acceptance still uses the exact ``H_tilde`` in both samplers, so the two
animations let you compare what the choice of integration Hamiltonian does to
the leapfrog arcs / acceptance.

We reuse the generic animation machinery (``run_chain`` + ``make_animation``)
from ``fhmc_illustration.py`` and define a small Rosenbrock sampler that mixes
the experiment's ``RosenbrockFHMCUnbiased`` (its U / K / H / H_tilde for the
Rosenbrock dual-Randers metric) with ``ImplicitFHMCUnbiased2Reg`` (the
H_tilde_reg integrator).

NOTE on H_tilde_reg: ``ImplicitFHMCUnbiased2Reg``'s own ``H_tilde_reg`` (in
samplers.py) omits the target potential ``U(z)`` -- that is correct for targets
where the Finsler volume form alone defines the density (e.g. the bump ring),
but here the Rosenbrock potential is a separate additive term (see
``RosenbrockFHMCUnbiased.H_tilde``). So this file overrides ``H_tilde_reg`` to
mirror that exact ``H_tilde`` with the Randers norm sqrt(p^T G* p) replaced by
the delta-smoothed sqrt(p^T G* p + delta) -- same U(z) / BH-volume terms,
smooth gradient near the Randers singularity.

NOTE: integrating H_tilde(_reg) adds the non-smooth log(1 + w*.p / ||p||_G*)
term to the dynamics, which makes the implicit fixed-point solves more prone to
divergence than the bare-H version; the delta-regularisation tempers but does
not remove this. We use a small step size here; run_chain also drops any
residual divergent arc (see DIVERGENCE_BOUND) instead of drawing the blow-up.

Run:
    python animations/fhmc2_illustration.py
Output:
    animations/fhmc2_illustration.mp4   (falls back to .gif if ffmpeg is missing)
"""

from __future__ import annotations

import os
import sys

import torch
from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.join(os.path.dirname(_HERE), "internship_experiments")
for _p in (_HERE, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from geodesic_toolbox import ImplicitFHMCUnbiased2Reg, RosenbrockDualRanders  # noqa: E402

from fhmc_illustration import run_chain, make_animation  # generic machinery  # noqa: E402
from rosenbrock_benchmark2 import RosenbrockFHMCUnbiased  # U / K / H / H_tilde  # noqa: E402


class RosenbrockFHMCUnbiased2Reg(ImplicitFHMCUnbiased2Reg):
    """Rosenbrock FHMC whose implicit leapfrog integrates a delta-regularised
    H_tilde (H_tilde_reg).

    Single-inheritance twin of ``RosenbrockFHMCUnbiased``: it reuses that class's
    U / K / H / H_tilde (the Rosenbrock dual-Randers energies, brought in by the
    assignments below) for the exact acceptance step, and defines a matching
    ``H_tilde_reg`` -- same U(z) + BH-volume terms, but with the Randers norm
    sqrt(p^T G* p) replaced by sqrt(p^T G* p + delta) -- so the leapfrog's own
    gradient stays finite at p = 0 / near the Randers singularity.
    """

    # Rosenbrock dual-Randers energies, reused verbatim.
    U = RosenbrockFHMCUnbiased.U
    K = RosenbrockFHMCUnbiased.K
    H = RosenbrockFHMCUnbiased.H
    H_tilde = RosenbrockFHMCUnbiased.H_tilde

    def __init__(self, l: int, N_fx: int, gamma: float, N_run: int,
                 bounds: float = 1e3, std_0: float = 1., beta_0: float = 1.,
                 pbar: bool = False, skip_acceptance: bool = False,
                 reduced_flip: bool = True, alpha: float = 1., beta: float = 1.,
                 delta_rel: float = 0.05):
        randers_cometric = RosenbrockDualRanders(alpha=alpha, beta=beta)
        super().__init__(
            randers_cometric=randers_cometric,
            l=l, N_fx=N_fx, gamma=gamma, N_run=N_run,
            bounds=bounds, std_0=std_0, beta_0=beta_0,
            pbar=pbar, skip_acceptance=skip_acceptance,
            reduced_flip=reduced_flip, delta_rel=delta_rel,
        )

    def H_tilde_reg(self, z: Tensor, p: Tensor) -> Tensor:
        """Delta-regularised twin of H_tilde (see class docstring)."""
        d = z.shape[1]
        delta = (self.delta_rel ** 2) * d

        G_inv, w_star, G_star = self.randers_cometric._shared(z)  # ONE eigh
        L = torch.linalg.cholesky(G_star)
        y = torch.linalg.solve_triangular(
            L, w_star.unsqueeze(-1), upper=False
        ).squeeze(-1)
        alpha_s = 1.0 - torch.einsum("bi,bi->b", y, y)
        logdet_G_star = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)
        log_sigma_BH = -0.5 * logdet_G_star

        p_Ginv_p = torch.einsum("bi,bij,bj->b", p, G_inv, p)
        wstar_p = torch.einsum("bi,bi->b", w_star, p)
        # sqrt(p^T G* p + delta), using p^T G* p = (w*.p)^2 + p^T G^-1 p / alpha_s
        n_delta = torch.sqrt(wstar_p**2 + p_Ginv_p / alpha_s + delta)
        F_delta = n_delta + wstar_p

        return (
            self.U(z)
            + 0.5 * F_delta**2
            - (d + 1) * torch.log1p(wstar_p / n_delta)
            + log_sigma_BH
            + 0.5 * d * self.log2pi
        )


if __name__ == "__main__":
    # ── FHMC sampler parameters (edit freely) ────────────────────────────────
    # Integrating H_tilde (K_tilde) is stiffer than integrating H, so we keep a
    # small step / many fixed-point iterations for a stable, drawable arc.
    L = 60                           # leapfrog steps
    GAMMA = 0.03                     # step size (smaller than the bare-H twin)
    BETA = 0.4                       # Randers drift strength (Finsler asymmetry)
    N_FX = 30                        # implicit fixed-point iterations
    ALPHA = 1.0                      # SoftAbs base-metric parameter
    STD_0 = 1.0                      # momentum scale
    REDUCED_FLIP = True              # reduced momentum flip on rejection
    DELTA_REL = 0.05                 # H_tilde_reg smoothing (relative to sqrt(d))

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 0
    N_ITER = 22
    Z0 = torch.tensor([[-2.0, 4.0]])
    XLIM, YLIM = (-3.5, 3.5), (-1.0, 9.0)

    # ── animation timing ─────────────────────────────────────────────────────
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    sampler = RosenbrockFHMCUnbiased2Reg(
        l=L, N_fx=N_FX, gamma=GAMMA, N_run=1,
        beta=BETA, alpha=ALPHA, std_0=STD_0, reduced_flip=REDUCED_FLIP,
        delta_rel=DELTA_REL,
    )

    records = run_chain(sampler, Z0, n_iter=N_ITER, seed=SEED)

    out_path = os.path.join(_HERE, "fhmc2_illustration.mp4")
    make_animation(
        records, sampler.U, l=L, gamma=GAMMA, beta=BETA,
        xlim=XLIM, ylim=YLIM,
        out_path=out_path,
        title="Finsler HMC ($\\tilde K$ integrator, regularised) — Rosenbrock (ImplicitFHMCUnbiased2Reg)",
        reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS,
    )
