"""
chi-feng style illustration of *Finsler* Hamiltonian Monte Carlo (FHMC) on the
Rosenbrock target.

Reuses the exact sampler from the internship experiments,
`RosenbrockFHMCUnbiased` (internship_experiments/rosenbrock_benchmark2.py), an
implicit FHMC on the dual Randers metric of the Rosenbrock potential

        U(x, y) = (100 (y - x^2)^2 + (1 - x)^2) / 20 ,

at its tuned operating point (beta = 0.4, l = 10, gamma = 0.57), all set in
this file's __main__ block.

The defining feature of FHMC vs. RHMC: the Randers metric is a *Finsler* metric
with a drift term  w = beta * omega(z).  The initial-velocity distribution is
therefore an ellipse that is BOTH anisotropic (like RHMC) AND **decentered** --
its mean is shifted off the current point, biasing motion in a preferred
direction along the valley.  We draw the exact level lines of the Randers velocity
density  exp(-1/2 (sqrt(v^T G v) + v^T omega)^2)  at the current point -- its level
sets are decentered ellipses.

Run:
    python animations/fhmc_illustration.py
Output:
    animations/fhmc_illustration.mp4   (falls back to .gif if ffmpeg is missing)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch import Tensor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
from torch.linalg import LinAlgError

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.join(os.path.dirname(_HERE), "internship_experiments")
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from rosenbrock_benchmark2 import RosenbrockFHMCUnbiased  # noqa: E402

DIVERGENCE_BOUND = 1e3  # arcs leaving this box are integration blow-ups, not dynamics


# --------------------------------------------------------------------------- #
def run_chain(sampler, z0: Tensor, n_iter: int, seed: int = 0,
              arrow_velocity: str = "legendre"):
    """FHMC sample loop (with the directional momentum flip), recording per
    iteration the leapfrog arc, the accept/reject decision, and the empirical
    initial-velocity distribution (mean + covariance) -- which is *decentered*
    because of the Randers drift.

    arrow_velocity : which velocity the *main* (cyan) arrow (rec['vel0']) shows.
        "legendre"   -> v = reversed_legendre(p0) = dH/dp, the sampled Randers
                        velocity (consistent with the momentum ellipses).
        "trajectory" -> v = dH_dv(z, v_half), the leapfrog's actual initial
                        velocity (exactly the direction of the first arc step).
        These coincide for the bare-H FHMC (H = 1/2 F*^2), but differ for
        ImplicitFHMCUnbiased2 / ImplicitFHMCUnbiased2Reg, whose leapfrog
        integrates H_tilde (or its delta-regularised twin H_tilde_reg): the
        -(d+1) log1p(<w*,p>/|p|_G*) correction term's p-gradient can rotate the
        launch velocity far from the sampled Legendre velocity (measured up to
        ~150 deg on the bump ring -- an exact-H_tilde effect, not the delta
        smoothing). Both velocities are recorded (rec['vel_traj'] always holds
        the trajectory one), and make_animation draws the trajectory launch as
        a second arc-green arrow whenever it deviates, so the sampled draw and
        the actual departure direction are both visible.
    """
    torch.manual_seed(seed)
    z = z0.clone()
    dirs = torch.ones(z.shape[0], device=z.device)
    records = []
    for _ in range(n_iter):
        dir_used = float(dirs[0].item())  # +1 forward, -1 backward (after a flip)
        p0 = sampler.sample_momentum(z)
        # sampled Randers velocity (a draw from the displayed ellipses) ...
        vel_leg = sampler.reversed_legendre(p0, z)[0].cpu().numpy().copy()
        # ... and the leapfrog's actual launch velocity (first arc step direction)
        try:
            v_half = sampler.get_v_half(z, p0, dirs)
            vel_traj = sampler.dH_dv(z, v_half)[0].cpu().numpy().copy()
            if not np.isfinite(vel_traj).all():
                vel_traj = vel_leg.copy()
        except Exception:  # divergent half-kick: fall back to the sampled velocity
            vel_traj = vel_leg.copy()
        vel0 = vel_traj if arrow_velocity == "trajectory" else vel_leg

        # Randers metric (G, omega) at z: the initial-velocity density is
        #   exp(-1/2 (sqrt(v^T G v) + v^T omega)^2)
        # whose level lines are decentered ellipses (Finsler asymmetry).
        primal = sampler.randers_cometric.primal_randers
        M = primal.base_cometric.metric_tensor(z)
        if primal.base_cometric.is_diag:
            M = torch.diag_embed(M)
        w = primal.beta * primal.omega(z)
        G_np = M[0].cpu().numpy().copy()
        w_np = w[0].cpu().numpy().copy()

        try:
            arc_q, arc_p = sampler.leapfrog(z, p0, dirs, return_traj=True)
            # The implicit leapfrog can diverge (its fixed-point solves fail) and
            # return a huge/NaN arc whose alpha underflows to a finite 0. Drop any
            # such non-physical trajectory rather than drawing the blow-up.
            if not torch.isfinite(arc_q).all() or arc_q.abs().max() > DIVERGENCE_BOUND:
                raise LinAlgError("divergent leapfrog trajectory")
            z_l, p_l = arc_q[:, -1, :], arc_p[:, -1, :]
            alpha = sampler.get_alpha(z, p0, z_l, p_l)
            if not torch.isfinite(alpha).all():
                raise LinAlgError("non-finite Hamiltonian")
        except LinAlgError:
            arc_q = torch.stack([z, z], dim=1)
            z_l = z.clone()
            alpha = torch.zeros(z.shape[0], device=z.device)

        u = torch.rand_like(alpha)
        accept = u < alpha
        # reduced momentum flip (Sohl-Dickstein 2012) on rejection, as in sample()
        if (not bool(accept.item())) and sampler.reduced_flip:
            try:
                z_lf, p_lf = sampler.leapfrog(z, p0, -dirs)
                alpha_flip = sampler.get_alpha(z, p0, z_lf, p_lf)
            except LinAlgError:
                alpha_flip = torch.zeros_like(alpha)
            p_flip = (alpha_flip - alpha).clamp(min=0)
            flip = bool((u < alpha + p_flip).item())
        else:
            flip = not bool(accept.item())

        accepted = bool(accept.item())
        z_end = z_l if accepted else z
        records.append(
            dict(
                start=z[0].cpu().numpy().copy(),
                direction=dir_used,
                G=G_np,
                w=w_np,
                vel0=vel0,
                vel_traj=vel_traj,
                arc=arc_q[0].cpu().numpy().copy(),
                proposal=z_l[0].cpu().numpy().copy(),
                accepted=accepted,
                end=z_end[0].cpu().numpy().copy(),
            )
        )
        z = z_end.clone()
        if flip:
            dirs = -dirs
    return records


# --------------------------------------------------------------------------- #
def make_animation(records, U_fn, l, gamma, beta,  xlim, ylim, out_path, title,
                   reveal=18, hold=8, n_mom=12, fps=30, vel_scale=1.0,
                   xlabel="$x$", ylabel="$y$"):
    # vel_scale = 1.0 -> data units = velocity units (true Randers metric scale).
    # The level lines of the Randers velocity density
    #   exp(-1/2 (sqrt(v^T G v) + v^T omega)^2)
    # are exactly the F = c level sets of the Randers norm, which are *ellipses*:
    #   F(v) = c  <=>  (v - v_c)^T M (v - v_c) = c^2 s ,
    #   M = G - omega omega^T,  v_c = -c M^{-1} omega,  s = 1 + omega^T M^{-1} omega.
    # We draw them analytically (one Ellipse per level) -- exact at any scale, with
    # no contour-grid clipping. Each level has its own centre v_c, so they are
    # decentered and nested (the Finsler asymmetry).
    F_LEVELS = (0.75, 1.5, 2.25)

    # --- static background: target density ---------------------------------- #
    nx = ny = 420
    xs = np.linspace(*xlim, nx)
    ys = np.linspace(*ylim, ny)
    XX, YY = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([XX.ravel(), YY.ravel()], axis=1))
    with torch.no_grad():
        dens = torch.exp(-U_fn(grid)).cpu().numpy().reshape(ny, nx)

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.imshow(dens, extent=[*xlim, *ylim], origin="lower", cmap="afmhot",
              aspect="auto", interpolation="bilinear", vmin=0, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # --- dynamic artists ---------------------------------------------------- #
    (chain_line,) = ax.plot([], [], "-", color="#7fd1ff", lw=1.0, alpha=0.5, zorder=3)
    chain_pts = ax.scatter([], [], s=14, color="#7fd1ff", edgecolors="none",
                           alpha=0.8, zorder=4)
    (arc_line,) = ax.plot([], [], "-", color="#9aff9a", lw=1.8, alpha=0.95, zorder=6)
    (cur_pt,) = ax.plot([], [], "o", color="white", ms=8, mec="black", mew=1.0, zorder=9)
    (prop_pt,) = ax.plot([], [], "o", ms=11, mfc="none", mew=1.8, zorder=8)

    # Randers velocity-density level lines, one Ellipse per F level (nested,
    # decentered). Geometry is set per frame in _set_ellipses().
    ells = [Ellipse((0, 0), 0, 0, fill=False, color="#27e0e0",
                    lw=1.6 if i == 0 else 1.0, zorder=7)
            for i, _ in enumerate(F_LEVELS)]
    for e in ells:
        ax.add_patch(e)
    Q = ax.quiver([0.0], [0.0], [0.0], [0.0], color="#27e0e0", zorder=8,
                  angles="xy", scale_units="xy", scale=1.0, width=0.006, alpha=0.0)
    # second arrow: the leapfrog's actual launch velocity (dH_tilde/dp at v_half),
    # drawn in the arc's green so "where the arc departs" is visible even when it
    # peels away from the sampled (cyan) Randers velocity -- an FHMC2 effect.
    Q2 = ax.quiver([0.0], [0.0], [0.0], [0.0], color="#9aff9a", zorder=7,
                   angles="xy", scale_units="xy", scale=1.0, width=0.005, alpha=0.0)

    info = ax.text(
        0.02, 0.975, "", transform=ax.transAxes, va="top", ha="left",
        color="white", fontsize=10,
        bbox=dict(boxstyle="round", fc="black", alpha=0.5, ec="none"),
    )
    # forward / backward integration badge (FHMC flips direction on rejection)
    badge = ax.text(
        0.98, 0.975, "", transform=ax.transAxes, va="top", ha="right",
        fontsize=12, fontweight="bold", color="white",
        bbox=dict(boxstyle="round", fc="#1f6f1f", alpha=0.85, ec="none"),
    )

    accepted_x = [records[0]["start"][0]]
    accepted_y = [records[0]["start"][1]]
    frames_per_iter = n_mom + reveal + hold
    total_frames = len(records) * frames_per_iter
    n_acc = [0]

    def _set_ellipses(start, G, w, alpha):
        # analytic Randers F = c level ellipses (see header comment)
        M = G - np.outer(w, w)
        Minv = np.linalg.inv(M)
        s = max(1.0 + float(w @ Minv @ w), 0.0)
        vals, vecs = np.linalg.eigh(M)
        vals = np.clip(vals, 1e-12, None)
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        Minv_w = Minv @ w
        for e, c in zip(ells, F_LEVELS):
            ctr = start + (-c * Minv_w) * vel_scale     # decentered centre
            r = c * np.sqrt(s) * vel_scale              # radius factor for level c
            e.set_center((ctr[0], ctr[1]))
            e.set_width(2 * r / np.sqrt(vals[0]))
            e.set_height(2 * r / np.sqrt(vals[1]))
            e.set_angle(ang)
            e.set_alpha(alpha)

    def init():
        for a in (chain_line, arc_line, cur_pt, prop_pt):
            a.set_data([], [])
        return [chain_line, chain_pts, arc_line, cur_pt, prop_pt, *ells, Q, Q2, info, badge]

    def update(frame):
        k = frame // frames_per_iter
        s = frame % frames_per_iter
        rec = records[k]
        arc = rec["arc"]
        n_steps = arc.shape[0]
        start, vel0, direction = rec["start"], rec["vel0"], rec["direction"]
        uv = vel0 * vel_scale * direction     # arrow points along actual travel
        vel_traj = rec.get("vel_traj", vel0)
        uv2 = vel_traj * vel_scale * direction  # actual leapfrog launch direction

        forward = direction > 0
        badge.set_text("▶  FORWARD" if forward else "◀  BACKWARD")
        badge.get_bbox_patch().set_facecolor("#1f6f1f" if forward else "#9c5a16")

        if s < n_mom:
            # phase 1: sample momentum -> reveal the decentered velocity density
            frac = (s + 1) / n_mom
            dens_alpha = 0.3 + 0.6 * frac
            arc_line.set_data([], [])
            prop_pt.set_data([], [])
            Q.set_offsets([[start[0], start[1]]])
            Q.set_UVC(uv[0] * frac, uv[1] * frac)
            Q.set_alpha(0.95)
            Q2.set_offsets([[start[0], start[1]]])
            Q2.set_UVC(uv2[0] * frac, uv2[1] * frac)
            Q2.set_alpha(0.85)
            cur_pt.set_data([start[0]], [start[1]])
            phase = "sample momentum  (Randers: decentered)"
        elif s < n_mom + reveal:
            # phase 2: implicit Finsler leapfrog
            ss = s - n_mom
            n_show = max(2, int(round((ss + 1) / reveal * n_steps)))
            dens_alpha = 0.45
            arc_line.set_data(arc[:n_show, 0], arc[:n_show, 1])
            arc_line.set_color("#9aff9a")
            prop_pt.set_data([], [])
            Q.set_offsets([[start[0], start[1]]])
            Q.set_UVC(uv[0], uv[1])
            Q.set_alpha(0.25)
            Q2.set_offsets([[start[0], start[1]]])
            Q2.set_UVC(uv2[0], uv2[1])
            Q2.set_alpha(0.25)
            cur_pt.set_data([start[0]], [start[1]])
            phase = "Finsler leapfrog"
        else:
            # phase 3: accept / reject
            col = "#3dff3d" if rec["accepted"] else "#ff5b5b"
            dens_alpha = 0.2
            arc_line.set_data(arc[:, 0], arc[:, 1])
            arc_line.set_color(col)
            prop_pt.set_data([rec["proposal"][0]], [rec["proposal"][1]])
            prop_pt.set_color(col)
            Q.set_alpha(0.0)
            Q2.set_alpha(0.0)
            phase = "ACCEPT" if rec["accepted"] else "REJECT"
            if s == n_mom + reveal:
                accepted_x.append(rec["end"][0])
                accepted_y.append(rec["end"][1])
                if rec["accepted"]:
                    n_acc[0] += 1

        # When integrating backward (direction = -1) the initial travel velocity
        # is -v with v ~ Randers(G, w); its density is the *reflected* Randers
        # density Randers(G, -w), so flip w with the direction. This keeps the
        # displayed ellipses consistent with the direction-multiplied arrows
        # (M = G - w w^T is unchanged; only the decentering flips side).
        _set_ellipses(start, rec["G"], rec["w"] * direction, dens_alpha)

        chain_line.set_data(accepted_x, accepted_y)
        chain_pts.set_offsets(np.column_stack([accepted_x, accepted_y]))
        if s >= n_mom + reveal:
            cur_pt.set_data([accepted_x[-1]], [accepted_y[-1]])

        committed = len(accepted_x) - 1
        rate = n_acc[0] / max(committed, 1)
        info.set_text(
            f"iteration {k + 1}/{len(records)}\n"
            f"leapfrog steps L = {l},  $\\gamma$ = {gamma}, $\\beta$ = {beta}\n"
            f"direction: {'forward' if forward else 'backward'}\n"
            f"phase: {phase}\n"
            f"accept rate: {rate:0.2f}"
        )
        return [chain_line, chain_pts, arc_line, cur_pt, prop_pt, *ells, Q, Q2, info, badge]

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, init_func=init,
        interval=1000 / fps, blit=False,
    )
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(out_path, writer=writer, dpi=120)
        print(f"saved {out_path}")
    except Exception as e:  # noqa: BLE001
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        print(f"ffmpeg failed ({e}); falling back to {gif_path}")
        anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=90)
        print(f"saved {gif_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # ── FHMC sampler parameters (edit freely) ────────────────────────────────
    # NOTE: the experiment's tuned point (l=10, gamma=0.57, N_fx=6) is tuned for
    # sampling efficiency, not accurate trajectories. At that step size the
    # implicit leapfrog on the stiff, nonlinear Randers/SoftAbs Hamiltonian does
    # not converge and the integrator diverges (arcs blow up to ~1e40+). FHMC is
    # more nonlinear than RHMC, so it needs an even smaller step. We integrate the
    # same dynamics far more accurately here; run_chain also drops any residual
    # divergent arc (see DIVERGENCE_BOUND) instead of drawing the blow-up.
    L = 60                           # leapfrog steps
    GAMMA = 0.05                     # step size
    BETA = 0.4                       # Randers drift strength (Finsler asymmetry)
    N_FX = 30                        # implicit fixed-point iterations
    ALPHA = 1.0                      # SoftAbs base-metric parameter
    STD_0 = 1.0                      # momentum scale
    REDUCED_FLIP = True            # reduced momentum flip on rejection

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 0
    N_ITER = 22
    Z0 = torch.tensor([[-2.0, 4.0]])
    XLIM, YLIM = (-3.5, 3.5), (-1.0, 9.0)

    # ── animation timing ─────────────────────────────────────────────────────
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    sampler = RosenbrockFHMCUnbiased(
        l=L, N_fx=N_FX, gamma=GAMMA, N_run=1,
        beta=BETA, alpha=ALPHA, std_0=STD_0, reduced_flip=REDUCED_FLIP,
    )

    records = run_chain(sampler, Z0, n_iter=N_ITER, seed=SEED)

    out_path = os.path.join(_HERE, "fhmc_illustration.mp4")
    make_animation(
        records, sampler.U, l=L, gamma=GAMMA, beta = BETA, 
        xlim=XLIM, ylim=YLIM,
        out_path=out_path,
        title="Finsler HMC — Rosenbrock target (RosenbrockFHMCUnbiased)",
        reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS,
    )
