"""
Shared utilities for the sampler illustration animations.

Collects the machinery previously duplicated across the individual
*_illustration.py scripts:

* run_chain_fhmc  - single-chain rollout for FHMC and FHMC_initial
                        (new integrator API), recording the Randers velocity
                        ellipse geometry for make_animation.
* run_chain_hmc / run_chain_rhmc - rollouts for the Euclidean HMC and
                        Riemannian HMC samplers.
* make_animation     - renderer with the decentered Randers velocity
                           ellipses (FHMC family).
* make_animation_hmc - renderer with the covariance velocity ellipses
                           (HMC / RHMC).
* make_ring_U        - robust ring-target background for the bump metric.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
from torch.linalg import LinAlgError

# arcs leaving this box are integration blow-ups, not dynamics
DIVERGENCE_BOUND = 1e3


def run_chain_fhmc(sampler, z0: Tensor, n_iter: int, seed: int = 0):
    """Single-chain rollout for FHMC / FHMC_initial (with the directional flip),
    producing the `records` format make_animation expects. vel0 is the sampled
    Randers velocity (cyan arrow); vel_traj is the actual launch velocity (green
    arrow) -- the field's position block for FHMC, the inverse Legendre for
    FHMC_initial.
    """
    #torch.manual_seed(seed)
    z = z0.clone()
    d = z.shape[1]
    dirs = torch.ones(z.shape[0], device=z.device)
    records = []
    for _ in range(n_iter):
        dir_used = float(dirs[0].item())  # +1 forward, -1 backward (after a flip)
        p0 = sampler.sample_momentum(z)
        x0 = torch.cat([z, p0], dim=-1)
        # sampled Randers velocity (cyan) and actual launch velocity (green)
        vel_leg = sampler.momentum_sampler.reversed_legendre(p0, z)[0].cpu().numpy().copy()
        try:
            if hasattr(sampler, "f"):
                vel_traj = sampler.f(x0)[0, :d].cpu().numpy().copy()
            else:  # FHMC_initial: dH/dp = inverse Legendre L*(p)
                vel_traj = sampler.momentum_sampler.reversed_legendre(p0, z)[0].cpu().numpy().copy()
            if not np.isfinite(vel_traj).all():
                vel_traj = vel_leg.copy()
        except Exception:  # noqa: BLE001
            vel_traj = vel_leg.copy()

        # Randers metric (G, omega) at z for the decentered velocity ellipses
        primal = sampler.randers_cometric.primal_randers
        M = primal.base_cometric.metric_tensor(z)
        if primal.base_cometric.is_diag:
            M = torch.diag_embed(M)
        w = primal.beta * primal.omega(z)
        G_np = M[0].cpu().numpy().copy()
        w_np = w[0].cpu().numpy().copy()

        try:
            traj_x, log_det = sampler.integrator(x0, return_traj=True, dirs=dirs)
            arc_q = traj_x[..., :d]
            if not torch.isfinite(arc_q).all() or arc_q.abs().max() > DIVERGENCE_BOUND:
                raise LinAlgError("divergent midpoint trajectory")
            x_l = traj_x[:, -1, :]
            alpha = sampler.proposal_rate(x0, x_l, log_det)  # NaN energies -> 0
        except LinAlgError:
            arc_q = torch.stack([z, z], dim=1)
            x_l = x0.clone()
            alpha = torch.zeros(z.shape[0], device=z.device)
        z_l = x_l[:, :d]

        u = torch.rand_like(alpha)
        accept = u < alpha
        # reduced flip (Sohl-Dickstein 2012) on rejection, as in sample()
        if (not bool(accept.item())) and sampler.reduced_flip:
            try:
                x_lf, log_det_f = sampler.integrator(x0, dirs=-dirs)
                alpha_flip = sampler.proposal_rate(x0, x_lf, log_det_f)
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
                vel0=vel_leg,
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
def run_chain_hmc(sampler, z0: Tensor, n_iter: int, seed: int = 0):
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
def run_chain_rhmc(sampler, z0: Tensor, n_iter: int, seed: int = 0):
    """HMCSampler.sample's loop, recording per iteration the leapfrog arc, the
    accept/reject decision, and the geometry of the velocity draw:
        momentum   p   ~ N(0, g(z))
        velocity   q'  = g(z)^{-1} p ~ N(0, g(z)^{-1})   <- the ellipse we draw
    """
    torch.manual_seed(seed)
    z = z0.clone()
    records = []
    for _ in range(n_iter):
        p0 = sampler.sample_momentum(z)
        g_inv = sampler.cometric.cometric_tensor(z)          # (1,2,2) = g(z)^{-1}
        cov = (g_inv[0] * sampler.std_0**2).cpu().numpy().copy()  # velocity cov
        vel0 = torch.einsum("bij,bj->bi", g_inv, p0)[0].cpu().numpy().copy()
        try:
            arc_q, arc_p = sampler.leapfrog(z, p0, return_traj=True)
            # drop a diverged arc rather than drawing the blow-up
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




# --------------------------------------------------------------------------- #
def make_animation(records, U_fn, l, gamma, beta,  xlim, ylim, out_path, title,
                   reveal=18, hold=8, n_mom=12, fps=30, vel_scale=1.0,
                   xlabel="$x$", ylabel="$y$"):
    # Draw the Randers velocity-density level lines F(v) = c as analytic ellipses
    # (M = G - omega omega^T, decentered by v_c = -c M^{-1} omega): nested and
    # off-centre, the Finsler asymmetry. vel_scale = 1 -> data units = velocity units.
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
    # second arrow (green): the actual launch velocity, so a departure that peels
    # away from the sampled (cyan) Randers velocity stays visible.
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

        # Backward integration reflects the velocity density (Randers(G, -w)),
        # so flip w with the direction to keep ellipses and arrows consistent.
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
def make_animation_hmc(records, U_fn, l, gamma, xlim, ylim, out_path, title,
                   reveal=18, hold=8, n_mom=12, fps=30, vel_scale=1.0,
                   xlabel="$x$", ylabel="$y$"):
    # Ellipses are the 1/2-sigma covariance of the velocity v ~ N(0, g(z)^{-1});
    # vel_scale = 1 means data units = velocity units.

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
            e.set_width(2 * ns * np.sqrt(vals[0]) * vel_scale)
            e.set_height(2 * ns * np.sqrt(vals[1]) * vel_scale)
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
        uv = vel0 * vel_scale

        if s < n_mom:
            # phase 1: sample momentum -> reveal the velocity ellipse + arrow
            frac = (s + 1) / n_mom
            _set_ellipses(start, cov, 0.25 + 0.65 * frac)
            arc_line.set_data([], [])
            prop_pt.set_data([], [])
            Q.set_offsets([[start[0], start[1]]])
            Q.set_UVC(uv[0] * frac, uv[1] * frac)
            Q.set_alpha(0.95)
            cur_pt.set_data([start[0]], [start[1]])
            phase = "sample momentum  $p\\sim\\mathcal{N}(0,g(z))$"
        elif s < n_mom + reveal:
            # phase 2: implicit leapfrog along the manifold
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
            phase = "Riemannian leapfrog"
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
