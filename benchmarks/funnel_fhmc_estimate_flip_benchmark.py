"""
Funnel / FHMC, ESTIMATED Jacobian, reduced momentum flip.
"""
import torch

import funnel_target as tgt
from benchmark_utils import run_benchmark

torch.set_default_dtype(torch.float64)

# Shared FHMC operating point; only reduced_flip and the Jacobian mode differs across the six variants.
PARAMS = {"beta": 0.10, "l": 20, "N_fx": 8, "gamma": 0.20, "alpha": 10 ** 6,
          "reg": 0.50, "method": "picard",
          "reduced_flip": True, "jacobian": "estimate",
          "jacobian_mc": 1, "russian_roulette": 0.9}
N_BATCH = 10       # chains, matching the RHMC / NUTS runners
N_RUN = 10000      # draws per chain, matching the RHMC / NUTS runners
SEED = 0           # RNG seed for the momentum draws (reproducibility)

if __name__ == "__main__":
    torch.manual_seed(SEED)
    z_0 = tgt.initial_states(N_BATCH, seed=777)
    sampler = tgt.build_sampler("FHMC", PARAMS, N_RUN)
    reference = tgt.reference_samples(400, seed=12345)
    run_benchmark(
        "FHMC_ESTIMATE_FLIP", sampler, z_0, PARAMS, target=tgt.NAME,
        score_fn=tgt.score, reference_samples=reference,
    )
