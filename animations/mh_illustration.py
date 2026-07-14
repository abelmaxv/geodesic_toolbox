"""
chi-feng style illustration of random-walk Metropolis-Hastings on the Rosenbrock
distribution.

Symmetric Gaussian proposal  z' ~ N(z, sigma^2 I), accepted with probability
min(1, pi(z')/pi(z)) = min(1, exp(-U(z') + U(z))), where the target is the same
Rosenbrock "banana" as the HMC/RHMC/FHMC scripts,

        pi(x, y) ∝ exp(-U),   U = (100 (y - x^2)^2 + (1 - x)^2) / 20 .

The proposal distribution is drawn as a fixed isotropic circle (1-sigma / 2-sigma)
around the current point -- it never adapts to the geometry, which is exactly why
random-walk MH mixes so slowly in the thin curved valley.

Run:
    python animations/mh_illustration.py
Output:
    animations/mh_illustration.mp4   (falls back to .gif if ffmpeg is missing)
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse

_HERE = os.path.dirname(os.path.abspath(__file__))


def rosenbrock_U(x, y):
    """Rosenbrock potential U = -log pi (same as the experiment's samplers)."""
    return (100.0 * (y - x**2) ** 2 + (1.0 - x) ** 2) / 20.0


# --------------------------------------------------------------------------- #
def run_chain(z0, n_iter, sigma, seed=0):
    """Random-walk Metropolis-Hastings loop, recording each proposal and the
    accept/reject decision for the animation."""
    rng = np.random.default_rng(seed)
    z = np.asarray(z0, dtype=float)
    records = []
    for _ in range(n_iter):
        prop = z + sigma * rng.standard_normal(2)
        log_ratio = -(rosenbrock_U(*prop) - rosenbrock_U(*z))  # log pi(z') - log pi(z)
        alpha = min(1.0, float(np.exp(min(log_ratio, 0.0))))
        accepted = rng.random() < alpha
        z_end = prop if accepted else z
        records.append(
            dict(start=z.copy(), proposal=prop.copy(),
                 accepted=accepted, end=z_end.copy())
        )
        z = z_end.copy()
    return records


# --------------------------------------------------------------------------- #
def make_animation(records, sigma, xlim, ylim, out_path,
                   reveal=8, hold=8, fps=30, xlabel="$x$", ylabel="$y$"):
    # --- static background: target density ---------------------------------- #
    nx = ny = 420
    xs = np.linspace(*xlim, nx)
    ys = np.linspace(*ylim, ny)
    XX, YY = np.meshgrid(xs, ys)
    dens = np.exp(-rosenbrock_U(XX, YY))

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
    (conn,) = ax.plot([], [], "--", color="#27e0e0", lw=1.0, alpha=0.8, zorder=5)
    (cur_pt,) = ax.plot([], [], "o", color="white", ms=8, mec="black", mew=1.0, zorder=9)
    (prop_pt,) = ax.plot([], [], "o", ms=10, mfc="none", mew=1.8, zorder=8)

    # proposal distribution N(z, sigma^2 I): fixed isotropic 1-/2-sigma circles
    c1 = Ellipse((0, 0), 0, 0, fill=False, color="#27e0e0", lw=1.6, zorder=7)
    c2 = Ellipse((0, 0), 0, 0, fill=False, color="#27e0e0", lw=1.0, alpha=0.5, zorder=7)
    ax.add_patch(c1)
    ax.add_patch(c2)

    info = ax.text(
        0.02, 0.975, "", transform=ax.transAxes, va="top", ha="left",
        color="white", fontsize=10,
        bbox=dict(boxstyle="round", fc="black", alpha=0.5, ec="none"),
    )

    accepted_x = [records[0]["start"][0]]
    accepted_y = [records[0]["start"][1]]
    frames_per_iter = reveal + hold
    total_frames = len(records) * frames_per_iter
    n_acc = [0]

    def _set_circles(center, alpha):
        for c, ns in ((c1, 1.0), (c2, 2.0)):
            c.set_center((center[0], center[1]))
            c.set_width(2 * ns * sigma)
            c.set_height(2 * ns * sigma)
        c1.set_alpha(alpha)
        c2.set_alpha(alpha * 0.55)

    def init():
        for a in (chain_line, conn, cur_pt, prop_pt):
            a.set_data([], [])
        return [chain_line, chain_pts, conn, cur_pt, prop_pt, c1, c2, info]

    def update(frame):
        k = frame // frames_per_iter
        s = frame % frames_per_iter
        rec = records[k]
        start, prop = rec["start"], rec["proposal"]

        cur_pt.set_data([start[0]], [start[1]])

        if s < reveal:
            # phase 1: draw proposal distribution + sampled proposal point
            frac = (s + 1) / reveal
            _set_circles(start, 0.25 + 0.65 * frac)
            prop_pt.set_data([prop[0]], [prop[1]])
            prop_pt.set_color("#27e0e0")
            conn.set_data([start[0], prop[0]], [start[1], prop[1]])
            conn.set_color("#27e0e0")
            phase = "propose  $z'\\sim\\mathcal{N}(z,\\sigma^2 I)$"
        else:
            # phase 2: accept / reject
            col = "#3dff3d" if rec["accepted"] else "#ff5b5b"
            _set_circles(start, 0.12)
            prop_pt.set_data([prop[0]], [prop[1]])
            prop_pt.set_color(col)
            conn.set_data([start[0], prop[0]], [start[1], prop[1]])
            conn.set_color(col)
            phase = "ACCEPT" if rec["accepted"] else "REJECT"
            if s == reveal:
                accepted_x.append(rec["end"][0])
                accepted_y.append(rec["end"][1])
                if rec["accepted"]:
                    n_acc[0] += 1
            cur_pt.set_data([accepted_x[-1]], [accepted_y[-1]])

        chain_line.set_data(accepted_x, accepted_y)
        chain_pts.set_offsets(np.column_stack([accepted_x, accepted_y]))

        committed = len(accepted_x) - 1
        rate = n_acc[0] / max(committed, 1)
        info.set_text(
            f"iteration {k + 1}/{len(records)}\n"
            f"proposal $\\sigma$ = {sigma}\n"
            f"phase: {phase}\n"
            f"accept rate: {rate:0.2f}"
        )
        return [chain_line, chain_pts, conn, cur_pt, prop_pt, c1, c2, info]

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
    # ── MH parameters (edit freely) ──────────────────────────────────────────
    SIGMA = 0.3      # symmetric Gaussian proposal standard deviation

    # ── chain / view ─────────────────────────────────────────────────────────
    SEED = 1
    N_ITER = 80
    Z0 = [-1.0, 1.0]
    XLIM, YLIM = (-2.5, 2.5), (-0.7, 5.0)

    # ── animation timing ─────────────────────────────────────────────────────
    REVEAL, HOLD, FPS = 7, 7, 30

    records = run_chain(Z0, N_ITER, SIGMA, seed=SEED)

    out_path = os.path.join(_HERE, "mh_illustration.mp4")
    make_animation(
        records, SIGMA, xlim=XLIM, ylim=YLIM, out_path=out_path,
        reveal=REVEAL, hold=HOLD, fps=FPS,
    )
