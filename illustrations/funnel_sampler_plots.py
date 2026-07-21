"""
Trajectory scatter plots of every sampler on Neal's funnel, rendered in the
cyan->magenta "chain scatter over the magma log-density" style of
funnel_sampling.ipynb, but using the sampler hyperparameters fixed in
funnel_benchmark2.py.

The sampler classes and their operating points are imported directly from
funnel_benchmark2 (``PARAMS`` / ``ALGORITHMS``), so the figures always reflect
the same hyperparameters as the benchmark. Only the plotting knobs below
(``N_TRAJ``, ``N_RUN``) are local -- they control how many chains and steps are
drawn, not the benchmark's diagnostics.

Each panel scatters ``N_TRAJ`` chains; the points of every chain are coloured
cyan->magenta along the chain index (time). The background is the marginal
log-density of (v, theta_1).

    cd internship_experiments && python funnel_sampler_plots.py

NOTE: the benchmark2 parameters use long trajectories (e.g. l=40 for RHMC/FHMC),
so sampling is slow. Lower N_TRAJ / N_RUN for a quicker preview.
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))
import funnel_target as tgt  # noqa: E402

STATE_DIM = tgt.STATE_DIM

# Sampler operating points (subset of the kept methods; edit freely). Keys map to
# funnel_target.build_sampler via BUILD_KEY.
PARAMS = {
    "HMC":          {"mass": 1, "l": 30, "gamma": 0.03},
    "RHMC":         {"l": 20, "N_fx": 25, "gamma": 0.2},
    "FHMC":         {"beta": 0.25, "l": 20, "N_fx": 8, "gamma": 0.01,
                     "alpha": 100.0, "reg": 0.05, "method": "picard", "reduced_flip": True},
    "FHMC_initial": {"beta": 0.25, "l": 20, "N_fx": 8, "gamma": 0.01,
                     "alpha": 100.0, "reduced_flip": True},
}
BUILD_KEY = {"HMC": "HMC", "RHMC": "RHMC", "FHMC": "FHMC", "FHMC_initial": "FHMC_INITIAL"}

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*torch.float32.*")


# ── Plot-only knobs (sampler hyperparameters come from funnel_benchmark2.PARAMS) ─
N_TRAJ         = 40                # chains drawn per panel
N_RUN          = 1000              # MCMC steps per chain (full trajectory length)
SNAPSHOT_STEPS = [100, 500, 1000]  # step counts at which to save an image
SEED           = 0
EXTENT         = 8.0                         # half-width of the (theta_1, v) plotting window

# cyan -> magenta gradient used to colour the chain samples along time
CHAIN_CMAP = LinearSegmentedColormap.from_list("cyan_magenta", ["#00e5ff", "#ff00d4"])


def funnel_log_pi_grid(grid_fine: int = 400, extent: float = EXTENT):
    """Marginal log-density of (v, theta_1) on a square grid (only one theta
    contributes the -v/2 term)."""
    t1 = np.linspace(-extent, extent, grid_fine)   # theta_1 axis (x)
    v  = np.linspace(-extent, extent, grid_fine)   # v axis      (y)
    tt, vv = np.meshgrid(t1, v)
    log_pi = -vv**2 / 18 - 0.5 * vv - np.exp(-vv) / 2 * tt**2
    return t1, v, log_pi


def scatter_funnel_chain(ax, traj, s: float = 4, alpha: float = 0.7):
    """Scatter funnel chains over the density background, coloured cyan->magenta
    along the chain index. Funnel layout: x = theta_1 (col 1), y = v (col 0)."""
    t = traj.detach().cpu().numpy() if hasattr(traj, "detach") else np.asarray(traj)
    colors = CHAIN_CMAP(np.linspace(0, 1, t.shape[1]))
    for i in range(t.shape[0]):
        ax.scatter(t[i, :, 1], t[i, :, 0], c=colors, s=s, alpha=alpha,
                   edgecolors="none", zorder=3)
    # start points
    ax.scatter(t[:, 0, 1], t[:, 0, 0], facecolor="white", edgecolor="black",
               linewidth=0.6, s=30, zorder=4)


def plot_panel(ax, traj, t1, v, log_pi):
    vmax = np.percentile(log_pi, 99)
    vmin = np.percentile(log_pi, 60)
    ax.imshow(log_pi, origin="lower",
              extent=[t1.min(), t1.max(), v.min(), v.max()],
              cmap="magma", aspect="auto", vmin=vmin, vmax=vmax, zorder=0)
    scatter_funnel_chain(ax, traj)
    ax.grid(True, linestyle=":", linewidth=0.5, color="white", alpha=0.2)
    ax.set_axisbelow(False)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$v$")
    ax.set_xlim(t1.min(), t1.max())
    ax.set_ylim(v.min(), v.max())


def param_tag(params: dict) -> str:
    """Filesystem-friendly summary of a sampler's hyperparameters."""
    return "_".join(f"{k}{v}" for k, v in params.items())


def sample_trajectories(name: str, n_traj: int = N_TRAJ, n_run: int = N_RUN,
                        seed: int = SEED, progress: bool = True):
    """Build the named sampler at its funnel_benchmark2 operating point and draw
    `n_traj` chains of `n_run` steps from the funnel mouth (z_0 = 0)."""
    torch.manual_seed(seed)
    sampler = tgt.build_sampler(BUILD_KEY[name], PARAMS[name], n_run)
    z_0 = torch.zeros(n_traj, STATE_DIM)
    if hasattr(sampler, "reduced_flip"):
        traj, acc, flip = sampler.sample(
            z_0, return_traj=True, progress=progress,
            return_acceptance=True, return_flip=True,
        )
    else:
        traj, acc = sampler.sample(
            z_0, return_traj=True, progress=progress, return_acceptance=True,
        )
        flip = None
    return traj, acc, flip


def _sample_worker(args: tuple):
    """Run one sampler to completion in its own process and return its full
    trajectory. Module-level with picklable args/returns so it ships cleanly to
    a ProcessPoolExecutor worker. Config is passed in explicitly (rather than
    read from module globals) so it is independent of how the child re-imports
    the module. Progress bars are off to avoid garbled concurrent output."""
    name, n_traj, n_run, seed = args
    traj, acc, flip = sample_trajectories(name, n_traj, n_run, seed, progress=False)
    return name, traj, acc, flip


def save_snapshot(traj, k, name, t1, v, log_pi, out_dir, timestamp):
    """Render and save one untitled snapshot of the chain up to step k.

    The filename encodes the sampler, its hyperparameters, the number of chains
    and the step count (k of N_RUN), e.g.
    ``funnel_FHMC_beta0.25_l40_N_fx6_gamma0.05_Ntraj40_step0200of0500_<ts>.png``.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_panel(ax, traj[:, :k + 1], t1, v, log_pi)   # z_0 (index 0) + first k samples
    fig.tight_layout()
    fname = (f"funnel_{name}_{param_tag(PARAMS[name])}"
             f"_Ntraj{N_TRAJ}_step{k:04d}of{N_RUN}_{timestamp}.png")
    out = os.path.join(out_dir, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    t1, v, log_pi = funnel_log_pi_grid()
    names = list(PARAMS.keys())

    out_dir = os.path.join("results", "funnel", "sampler_plots")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    steps = [k for k in SNAPSHOT_STEPS if k <= N_RUN]

    print(f"{'='*55}")
    print("  FUNNEL SAMPLER PLOTS  (funnel_benchmark2 parameters)")
    print(f"{'='*55}")
    print(f"  N_traj={N_TRAJ}  N_run={N_RUN}  snapshots={steps}  state_dim={STATE_DIM}")
    for name in names:
        print(f"  {name:15s} : {PARAMS[name]}")

    # Sample all four samplers in parallel: each is internally vectorized over
    # its N_TRAJ chains, and the four are independent -> one process each.
    print(f"\nSampling {len(names)} samplers in parallel ...", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=len(names)) as ex:
        futures = [ex.submit(_sample_worker, (name, N_TRAJ, N_RUN, SEED)) for name in names]
        for fut in as_completed(futures):
            name, traj, acc, flip = fut.result()
            results[name] = (traj, acc, flip)
            flip_str = f", flip={flip:.3f}" if flip is not None else ""
            print(f"  done: {name} (acc={acc:.3f}{flip_str})", flush=True)

    # Plotting stays on the main process; save one untitled image per requested
    # step count, showing the chain accumulated up to that step.
    print("\nSaving snapshots ...")
    for name in names:
        traj, _, _ = results[name]
        for k in steps:
            out = save_snapshot(traj, k, name, t1, v, log_pi, out_dir, timestamp)
            print(f"  {name:15s} step {k:>5}/{N_RUN}  ->  {out}")


if __name__ == "__main__":
    main()
