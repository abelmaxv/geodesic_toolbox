"""
Funnel / RHMC with the implicit midpoint integrator.

Reproduces Brofos & Lederman (ICML 2021, arXiv:2102.07139) table 2, row
I.M.(a), step-size 0.5 -- their best result: acc 0.85, min ESS 10147/10000.
Config matches their code (github.com/JamesBrofos/Evaluating-the-Implicit-
Midpoint-Integrator). NB their min ESS exceeds the draw count; our Geyer
estimator caps at N, so read our min-ESS as a lower bound and compare on
acceptance.

"""
import torch

import funnel_target as tgt
from benchmark_utils import run_benchmark

torch.set_default_dtype(torch.float64)

# Their table 2 operating point; threshold_fx is their fixed-point tolerance.
PARAMS = {"l": 20, "gamma": 0.5, "alpha": 10 ** 6, "N_fx": 50,
          "threshold_fx": 1e-6, "reduced_flip": False}
N_BATCH = 10       # chains ~ their 10 trials
N_RUN = 10000      # draws per chain, as the paper
SEED = 0          # RNG seed for the momentum draws (reproducibility)

if __name__ == "__main__":
    torch.manual_seed(SEED)
    z_0 = tgt.initial_states(N_BATCH, seed=777)
    sampler = tgt.build_sampler("RHMC_MIDPOINT", PARAMS, N_RUN)
    reference = tgt.reference_samples(400, seed=12345)
    run_benchmark(
        "RHMC_MIDPOINT", sampler, z_0, PARAMS, target=tgt.NAME,
        score_fn=tgt.score, reference_samples=reference,
    )
