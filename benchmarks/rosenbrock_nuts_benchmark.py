"""
Rosenbrock / NUTS benchmark -- one fixed configuration.
NUTS is self-tuning (dual-averaging); only warm-up / target-acceptance are set.

    cd benchmarks && python rosenbrock_nuts_benchmark.py
"""
import torch

import rosenbrock_target as tgt
from benchmark_utils import run_benchmark

torch.set_default_dtype(torch.float64)

# Fixed operating point (edit to change the configuration).
PARAMS = {
    "burn": 200,          # warm-up samples discarded per chain
    "init_step": 0.05,    # initial leapfrog step (dual-averaging adapts it)
    "max_steps": 1024,    # max tree depth (2^n leapfrog steps) per sample
    "accept": 0.8,        # Stan-default target acceptance for the adaptation
}
N_BATCH = 20      # parallel chains
N_RUN = 1000      # samples per chain

if __name__ == "__main__":
    z_0 = tgt.initial_states(N_BATCH, seed=777)
    sampler = tgt.build_sampler("NUTS", PARAMS, N_RUN)
    reference = tgt.reference_samples(400, seed=12345) if hasattr(tgt, "reference_samples") else None
    run_benchmark(
        "NUTS", sampler, z_0, PARAMS, target=tgt.NAME,
        score_fn=tgt.score, reference_samples=reference,
    )
