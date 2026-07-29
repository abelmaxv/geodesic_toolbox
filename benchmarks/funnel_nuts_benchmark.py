"""
Funnel / NUTS benchmark 

"""
import torch

import funnel_target as tgt
from benchmark_utils import run_benchmark

torch.set_default_dtype(torch.float64)

# NUTS is self-tuning (dual-averaging); only warm-up / target-acceptance are set.
PARAMS = {
    "burn": 200,          # warm-up samples discarded per chain
    "init_step": 0.05,    # initial leapfrog step (dual-averaging adapts it)
    "max_steps": 1024,    # max tree depth (2^n leapfrog steps) per sample
    "accept": 0.8,        # Stan-default target acceptance for the adaptation
}
N_BATCH = 10       # chains ~ the RHMC reproduction's 10 trials
N_RUN = 10000      # draws per chain (matches the RHMC reproduction)
SEED = 0          # RNG seed for the momentum draws (reproducibility)

if __name__ == "__main__":
    torch.manual_seed(SEED)
    z_0 = tgt.initial_states(N_BATCH, seed=777)
    sampler = tgt.build_sampler("NUTS", PARAMS, N_RUN)
    reference = tgt.reference_samples(400, seed=12345)
    run_benchmark(
        "NUTS", sampler, z_0, PARAMS, target=tgt.NAME,
        score_fn=tgt.score, reference_samples=reference,
    )
