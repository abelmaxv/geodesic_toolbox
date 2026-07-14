"""
chi-feng style illustration of *Finsler* HMC on Neal's 2D funnel, using the
``ImplicitFHMCUnbiased2Reg`` variant (the leapfrog integrates a delta-
regularised corrected Randers Hamiltonian ``H_tilde_reg`` / ``K_tilde``
instead of the bare ``H`` / ``K``).

Twin of ``funnel_fhmc_illustration.py``: same target, view and animation
machinery, only the integration Hamiltonian differs, so the two can be compared
side by side. The 2D funnel is the dim=1 case, state z = (v, theta), with
potential

        U(v, theta) = v^2/18 + v/2 + exp(-v)/2 * theta^2 .

We reuse the generic machinery (``run_chain`` + ``make_animation``) from
``fhmc_illustration.py`` and define a small funnel sampler that mixes the
experiment's ``FunnelFHMCUnbiased`` (its U / K / H / H_tilde for the funnel
dual-Randers metric) with ``ImplicitFHMCUnbiased2Reg`` (the H_tilde_reg
integrator).

NOTE on H_tilde_reg: ``ImplicitFHMCUnbiased2Reg``'s own ``H_tilde_reg`` (in
samplers.py) omits the target potential ``U(z)`` -- correct for targets where
the Finsler volume form alone defines the density (e.g. the bump ring), but
here the funnel potential is a separate additive term (see
``FunnelFHMCUnbiased.H_tilde``). So this file overrides ``H_tilde_reg`` to
mirror that exact ``H_tilde`` with the Randers norm sqrt(p^T G* p) replaced by
the delta-smoothed sqrt(p^T G* p + delta) -- same U(z) / BH-volume terms,
smooth gradient near the Randers singularity.

NOTE on alpha: as in the other funnel scripts, the experiment's SoftAbs
alpha=1e6 is too stiff for the implicit leapfrog, so we soften it (alpha~10).
Integrating H_tilde(_reg) is stiffer still, so we also use a smaller step than
the bare-H twin; run_chain guards any residual divergence.

Run:
    python animations/funnel_fhmc2_illustration.py
Output:
    animations/funnel_fhmc2_illustration.mp4
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

from geodesic_toolbox import ImplicitFHMCUnbiased2Reg, FunnelDualRanders  # noqa: E402

from fhmc_illustration import run_chain, make_animation  # generic machinery  # noqa: E402
from funnel_benchmark2 import FunnelFHMCUnbiased  # U / K / H / H_tilde  # noqa: E402


class FunnelFHMCUnbiased2Reg(ImplicitFHMCUnbiased2Reg):
    """Funnel FHMC whose implicit leapfrog integrates a delta-regularised
    H_tilde (H_tilde_reg).

    Single-inheritance twin of ``FunnelFHMCUnbiased``: it reuses that class's
    U / K / H / H_tilde (the funnel dual-Randers energies, brought in by the
    assignments below) for the exact acceptance step, and defines a matching
    ``H_tilde_reg`` -- same U(z) + BH-volume terms, but with the Randers norm
    sqrt(p^T G* p) replaced by sqrt(p^T G* p + delta) -- so the leapfrog's own
    gradient stays finite at p = 0 / near the Randers singularity.
    """

    # Funnel dual-Randers energies, reused verbatim.
    U = FunnelFHMCUnbiased.U
    K = FunnelFHMCUnbiased.K
    H = FunnelFHMCUnbiased.H
    H_tilde = FunnelFHMCUnbiased.H_tilde

    def __init__(self, dim: int, l: int, N_fx: int, gamma: float, N_run: int,
                 bounds: float = 1e3, std_0: float = 1., beta_0: float = 1.,
                 pbar: bool = False, skip_acceptance: bool = False,
                 reduced_flip: bool = True, alpha: float = 10**6, beta: float = 1.,
                 delta_rel: float = 0.05):
        self.dim = dim  # U / H / H_tilde read self.dim
        randers_cometric = FunnelDualRanders(dim=dim, alpha=alpha, beta=beta)
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
    DIM = 1          # 2D funnel: state (v, theta)
    L = 20           # leapfrog steps
    GAMMA = 0.03     # step size (smaller than the bare-H twin: H_tilde is stiffer)
    N_FX = 12        # implicit fixed-point iterations
    BETA = 0.2       # Randers drift strength (Finsler asymmetry)
    ALPHA = 10.0     # SoftAbs sharpness (softened from the experiment's 1e6)
    STD_0 = 1.0      # momentum scale
    REDUCED_FLIP = False  # reduced momentum flip on rejection
    DELTA_REL = 0.05      # H_tilde_reg smoothing (relative to sqrt(d))

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 2
    N_ITER = 24
    Z0 = torch.tensor([[1.0, 0.0]])
    XLIM, YLIM = (-6.0, 5.0), (-7.0, 7.0)

    # ── ellipse / animation timing ───────────────────────────────────────────
    VEL_SCALE = 1.0  # 1.0 = true metric scale for the Randers density level lines
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    sampler = FunnelFHMCUnbiased2Reg(
        dim=DIM, l=L, N_fx=N_FX, gamma=GAMMA, N_run=1,
        beta=BETA, alpha=ALPHA, std_0=STD_0, reduced_flip=REDUCED_FLIP,
        delta_rel=DELTA_REL,
    )

    records = run_chain(sampler, Z0, n_iter=N_ITER, seed=SEED)

    out_path = os.path.join(_HERE, "funnel_fhmc2_illustration.mp4")
    make_animation(
        records, sampler.U, l=L, gamma=GAMMA, beta=BETA,
        xlim=XLIM, ylim=YLIM,
        out_path=out_path,
        title=None,
        reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS, vel_scale=VEL_SCALE,
        xlabel="$v$", ylabel=r"$\theta$",
    )
