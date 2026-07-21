"""
Run ONLY RHMC on the Rosenbrock target at a fixed operating point, reusing the
benchmarks/ machinery (rosenbrock_target + benchmark_utils) so the numbers are
comparable to the full benchmark suite.

    cd illustrations && python rosenbrock_rhmc_only.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))
import rosenbrock_target as tgt  # noqa: E402
from benchmark_utils import run_benchmark  # noqa: E402

torch.set_default_dtype(torch.float64)

PARAMS = {"l": 10, "N_fx": 6, "gamma": 0.07}
N_BATCH = 100
N_RUN = 2000

if __name__ == "__main__":
    z_0 = tgt.initial_states(N_BATCH, seed=777)
    sampler = tgt.build_sampler("RHMC", PARAMS, N_RUN)
    run_benchmark("RHMC", sampler, z_0, PARAMS, target=tgt.NAME, score_fn=tgt.score)
