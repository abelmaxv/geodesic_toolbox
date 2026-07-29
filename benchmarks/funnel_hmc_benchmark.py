"""
Funnel / Euclidean HMC benchmark 

"""
import torch

import funnel_target as tgt
from benchmark_utils import run_benchmark

torch.set_default_dtype(torch.float64)

# Fixed operating point (edit to change the configuration). T = gamma*l = 0.9.
PARAMS = {"mass": 1.0, "l": 30, "gamma": 0.03}
N_BATCH = 10       # chains, matching the RHMC / NUTS runners
N_RUN = 10000      # draws per chain, matching the RHMC / NUTS runners
SEED = 0          # RNG seed for the momentum draws (reproducibility)

if __name__ == "__main__":
    torch.manual_seed(SEED)
    z_0 = tgt.initial_states(N_BATCH, seed=777)
    sampler = tgt.build_sampler("HMC", PARAMS, N_RUN)
    reference = tgt.reference_samples(400, seed=12345)
    run_benchmark(
        "HMC", sampler, z_0, PARAMS, target=tgt.NAME,
        score_fn=tgt.score, reference_samples=reference,
    )
