"""
chi-feng style illustration of (Euclidean) Hamiltonian Monte Carlo on the
Rosenbrock target.

Reuses the exact sampler from the internship experiments, `RosenbrockHMC`
(internship_experiments/rosenbrock_benchmark2.py): a plain mass-matrix HMC
(HMCSampler + IdentityCoMetric) whose potential is the Rosenbrock "banana"

        U(x, y) = (100 (y - x^2)^2 + (1 - x)^2) / 20 ,

at its tuned operating point (mass = 20, l = 60, gamma = 0.05), all set in
this file's __main__ block.

The velocity distribution is drawn as an ellipse on the density at the current
point.  Because the mass matrix is constant (g(z) = I / mass), this ellipse is a
*fixed isotropic circle* everywhere -- contrast this with rhmc_illustration.py,
where the Riemannian metric makes it tilt and stretch with the local geometry.

Run:
    python animations/hmc_illustration.py
Output:
    animations/hmc_illustration.mp4   (falls back to .gif if ffmpeg is missing)
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
from rosenbrock_benchmark2 import RosenbrockHMC  # noqa: E402


# --------------------------------------------------------------------------- #
def run_chain(sampler, z0: Tensor, n_iter: int, seed: int = 0):
    """HMCSampler.sample's loop, recording per iteration the leapfrog arc, the
    accept/reject decision, and the geometry of the velocity draw.  For
    Euclidean HMC the momentum is the velocity, drawn from N(0, g(z)) with the
    constant mass g(z) = I / mass -> an isotropic circle."""
    torch.manual_seed(seed)
    z = z0.clone()
    records = []
    for _ in range(n_iter):
        p0 = sampler.sample_momentum(z)
        g = sampler.cometric.metric_tensor(z)                # (1,2,2) = mass matrix
        cov = (g[0] * sampler.std_0**2).cpu().numpy().copy()  # velocity cov
        vel0 = p0[0].cpu().numpy().copy()                    # v = p for Euclidean K
        try:
            arc_q, arc_p = sampler.leapfrog(z, p0, return_traj=True)
            z_l, p_l = arc_q[:, -1, :], arc_p[:, -1, :]
            alpha = sampler.get_alpha(z, p0, z_l, p_l)
            if not torch.isfinite(alpha).all():
                raise LinAlgError("non-finite Hamiltonian")
        except LinAlgError:
            arc_q = torch.stack([z, z], dim=1)
            z_l = z.clone()
            alpha = torch.zeros(z.shape[0], device=z.device)
        u = torch.rand_like(alpha)
        accepted = bool((alpha >= u).item())
        z_end = z_l if accepted else z
        records.append(
            dict(
                start=z[0].cpu().numpy().copy(),
                cov=cov,
                vel0=vel0,
                arc=arc_q[0].cpu().numpy().copy(),
                proposal=z_l[0].cpu().numpy().copy(),
                accepted=accepted,
                end=z_end[0].cpu().numpy().copy(),
            )
        )
        z = z_end.clone()
    return records


# --------------------------------------------------------------------------- #
def make_animation(records, U_fn, l, gamma, xlim, ylim, out_path, title,
                   reveal=18, hold=8, n_mom=12, fps=30, xlabel="$x$", ylabel="$y$"):
    scale = l * gamma  # free-flight displacement scale for the velocity ellipse

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

    # velocity-distribution ellipses (1 sigma, 2 sigma) + launch arrow
    e1 = Ellipse((0, 0), 0, 0, fill=False, color="#27e0e0", lw=1.6, zorder=7)
    e2 = Ellipse((0, 0), 0, 0, fill=False, color="#27e0e0", lw=1.0, alpha=0.5, zorder=7)
    ax.add_patch(e1)
    ax.add_patch(e2)
    Q = ax.quiver([0.0], [0.0], [0.0], [0.0], color="#27e0e0", zorder=8,
                  angles="xy", scale_units="xy", scale=1.0, width=0.006, alpha=0.0)

    info = ax.text(
        0.02, 0.975, "", transform=ax.transAxes, va="top", ha="left",
        color="white", fontsize=10,
        bbox=dict(boxstyle="round", fc="black", alpha=0.5, ec="none"),
    )

    accepted_x = [records[0]["start"][0]]
    accepted_y = [records[0]["start"][1]]
    frames_per_iter = n_mom + reveal + hold
    total_frames = len(records) * frames_per_iter
    n_acc = [0]

    def _set_ellipses(center, cov, alpha):
        vals, vecs = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        for e, ns in ((e1, 1.0), (e2, 2.0)):
            e.set_center((center[0], center[1]))
            e.set_width(2 * ns * np.sqrt(vals[0]) * scale)
            e.set_height(2 * ns * np.sqrt(vals[1]) * scale)
            e.set_angle(ang)
        e1.set_alpha(alpha)
        e2.set_alpha(alpha * 0.55)

    def init():
        for a in (chain_line, arc_line, cur_pt, prop_pt):
            a.set_data([], [])
        return [chain_line, chain_pts, arc_line, cur_pt, prop_pt, e1, e2, Q, info]

    def update(frame):
        k = frame // frames_per_iter
        s = frame % frames_per_iter
        rec = records[k]
        arc = rec["arc"]
        n_steps = arc.shape[0]
        start, cov, vel0 = rec["start"], rec["cov"], rec["vel0"]
        uv = vel0 * scale

        if s < n_mom:
            # phase 1: sample momentum -> reveal the velocity circle + arrow
            frac = (s + 1) / n_mom
            _set_ellipses(start, cov, 0.25 + 0.65 * frac)
            arc_line.set_data([], [])
            prop_pt.set_data([], [])
            Q.set_offsets([[start[0], start[1]]])
            Q.set_UVC(uv[0] * frac, uv[1] * frac)
            Q.set_alpha(0.95)
            cur_pt.set_data([start[0]], [start[1]])
            phase = "sample momentum  $p\\sim\\mathcal{N}(0,g)$"
        elif s < n_mom + reveal:
            # phase 2: leapfrog integration
            ss = s - n_mom
            n_show = max(2, int(round((ss + 1) / reveal * n_steps)))
            _set_ellipses(start, cov, 0.35)
            arc_line.set_data(arc[:n_show, 0], arc[:n_show, 1])
            arc_line.set_color("#9aff9a")
            prop_pt.set_data([], [])
            Q.set_offsets([[start[0], start[1]]])
            Q.set_UVC(uv[0], uv[1])
            Q.set_alpha(0.25)
            cur_pt.set_data([start[0]], [start[1]])
            phase = "leapfrog integration"
        else:
            # phase 3: accept / reject
            col = "#3dff3d" if rec["accepted"] else "#ff5b5b"
            _set_ellipses(start, cov, 0.12)
            arc_line.set_data(arc[:, 0], arc[:, 1])
            arc_line.set_color(col)
            prop_pt.set_data([rec["proposal"][0]], [rec["proposal"][1]])
            prop_pt.set_color(col)
            Q.set_alpha(0.0)
            phase = "ACCEPT" if rec["accepted"] else "REJECT"
            if s == n_mom + reveal:
                accepted_x.append(rec["end"][0])
                accepted_y.append(rec["end"][1])
                if rec["accepted"]:
                    n_acc[0] += 1

        chain_line.set_data(accepted_x, accepted_y)
        chain_pts.set_offsets(np.column_stack([accepted_x, accepted_y]))
        if s >= n_mom + reveal:
            cur_pt.set_data([accepted_x[-1]], [accepted_y[-1]])

        committed = len(accepted_x) - 1
        rate = n_acc[0] / max(committed, 1)
        info.set_text(
            f"iteration {k + 1}/{len(records)}\n"
            f"leapfrog steps L = {l},  $\\gamma$ = {gamma}\n"
            f"phase: {phase}\n"
            f"accept rate: {rate:0.2f}"
        )
        return [chain_line, chain_pts, arc_line, cur_pt, prop_pt, e1, e2, Q, info]

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
    # ── HMC sampler parameters (edit freely) ─────────────────────────────────
    # Defaults match the tuned operating point from rosenbrock_benchmark2.py.
    MASS = 20        # mass-matrix scale (constant metric g = I / MASS)
    L = 60           # leapfrog steps
    GAMMA = 0.05     # step size
    STD_0 = 1.0      # momentum scale
    BOUNDS = 1e3     # reject proposals with ||z|| beyond this

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 1
    N_ITER = 30
    Z0 = torch.tensor([[-1.5, 2.5]])
    XLIM, YLIM = (-2.5, 2.5), (-0.7, 5.0)

    # ── animation timing ─────────────────────────────────────────────────────
    N_MOM, REVEAL, HOLD, FPS = 12, 18, 8, 30

    sampler = RosenbrockHMC(
        mass=MASS, l=L, gamma=GAMMA,
        N_run=1, bounds=BOUNDS, std_0=STD_0,
    )

    records = run_chain(sampler, Z0, n_iter=N_ITER, seed=SEED)

    out_path = os.path.join(_HERE, "hmc_illustration.mp4")
    make_animation(
        records, sampler.U, l=L, gamma=GAMMA,
        xlim=XLIM, ylim=YLIM,
        out_path=out_path,
        title="Hamiltonian Monte Carlo — Rosenbrock target (RosenbrockHMC)",
        reveal=REVEAL, hold=HOLD, n_mom=N_MOM, fps=FPS,
    )
