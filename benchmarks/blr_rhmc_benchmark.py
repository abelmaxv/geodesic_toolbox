"""
Bayesian logistic regression / RHMC benchmark -- one fixed configuration.

    cd benchmarks && python blr_rhmc_benchmark.py
"""
import torch

import blr_target as tgt
from benchmark_utils import run_benchmark

torch.set_default_dtype(torch.float64)

# Fixed operating point (edit to change the configuration).
PARAMS = {"l": 10, "N_fx": 6, "gamma": 0.04}
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
