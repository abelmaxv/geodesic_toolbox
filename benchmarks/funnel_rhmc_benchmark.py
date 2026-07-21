"""
Funnel / RHMC benchmark -- one fixed configuration.
Brofos & Lederman (ICML 2021) Table 2 generalized-leapfrog point.

    cd benchmarks && python funnel_rhmc_benchmark.py
"""
import torch

import funnel_target as tgt
from benchmark_utils import run_benchmark

torch.set_default_dtype(torch.float64)

# Fixed operating point (edit to change the configuration).
PARAMS = {"l": 20, "N_fx": 25, "gamma": 0.2}
N_BATCH = 20      # parallel chains
N_RUN = 1000      # samples per chain

if __name__ == "__main__":
    z_0 = tgt.initial_states(N_BATCH, seed=777)
    sampler = tgt.build_sampler("RHMC", PARAMS, N_RUN)
    reference = tgt.reference_samples(400, seed=12345) if hasattr(tgt, "reference_samples") else None
    run_benchmark(
        "RHMC", sampler, z_0, PARAMS, target=tgt.NAME,
        score_fn=tgt.score, reference_samples=reference,
    )
