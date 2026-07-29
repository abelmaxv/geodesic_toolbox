"""
Bayesian logistic regression / FHMC_initial benchmark 
"""
import torch

import blr_target as tgt
from benchmark_utils import run_benchmark

torch.set_default_dtype(torch.float64)

# Fixed operating point (edit to change the configuration).
PARAMS = {"beta": 0.10, "l": 10, "N_fx": 6, "gamma": 0.07,
          "alpha": 1.0, "reduced_flip": True}
N_BATCH = 20      # parallel chains
N_RUN = 1000      # samples per chain
SEED = 0          # RNG seed for the momentum draws (reproducibility)

if __name__ == "__main__":
    torch.manual_seed(SEED)
    z_0 = tgt.initial_states(N_BATCH, seed=777)
    sampler = tgt.build_sampler("FHMC_INITIAL", PARAMS, N_RUN)
    reference = tgt.reference_samples(400, seed=12345) if hasattr(tgt, "reference_samples") else None
    run_benchmark(
        "FHMC_initial", sampler, z_0, PARAMS, target=tgt.NAME,
        score_fn=tgt.score, reference_samples=reference,
    )
