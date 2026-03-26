#!/usr/bin/env python3
"""Benchmark GAN quality using KS test across diverse HMM states and contexts."""
import json
import numpy as np
import click
from pathlib import Path
from loguru import logger

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# HMM state names
HMM_STATE_NAMES = ["NORMAL", "SLOW", "FAST", "ERROR", "CORRECTION", "PAUSE"]

# Natural occurrence weights (based on DEFAULT_TRANSMAT steady state)
HMM_STATE_WEIGHTS = [0.60, 0.15, 0.12, 0.05, 0.04, 0.04]


@click.command()
@click.option("--model-dir", default="models")
@click.option("--data", default="data/raw/keystroke_samples.jsonl")
def main(model_dir, data):
    if not SCIPY_AVAILABLE:
        logger.error("scipy required for benchmarking")
        return

    # Load real data
    real_delays = []
    path = Path(data)
    if path.exists():
        with open(path) as f:
            for line in f:
                record = json.loads(line.strip())
                timings = record.get("timings", [])
                for t in timings:
                    delay = t[0] if isinstance(t, list) else t
                    real_delays.append(delay)
    else:
        real_delays = np.random.lognormal(np.log(120), 0.5, 1000).tolist()
        logger.warning("No real data found, using synthetic reference")

    import yaml
    config = {}
    if Path("config.yaml").exists():
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

    from layer3_dynamics.gan.inference import GANInference
    gan = GANInference(
        model_path=f"{model_dir}/gan_generator.pth",
        config=config.get("gan", {}),
    )

    # ── Diverse sampling across all HMM states, complexity, and fatigue ──
    N_BASE = 16
    gen_delays = []
    state_medians = {}

    for hmm_state, (weight, name) in enumerate(zip(HMM_STATE_WEIGHTS, HMM_STATE_NAMES)):
        n = max(1, int(N_BASE * weight * len(HMM_STATE_WEIGHTS)))
        state_delays = []

        for complexity in [1, 2, 4]:          # SIMPLE / MODERATE / VERY_COMPLEX
            for fatigue in [1.0, 0.8, 0.6]:   # fresh / mid / tired
                ctx = gan.build_context_vector(
                    complexity=complexity,
                    fatigue=fatigue,
                    hmm_state=hmm_state,
                )
                timings = gan.sample_timings(ctx, n_samples=n)  # (n, seq_len, 3)
                delays = timings[:, :, 0].flatten().tolist()
                state_delays.extend(delays)

        gen_delays.extend(state_delays)
        state_medians[name] = np.median(state_delays)

    real_arr = np.array(real_delays)
    gen_arr = np.array(gen_delays)

    # ── Per-state breakdown ──
    print(f"\nHMM State Breakdown (GAN median ms):")
    for name, median in state_medians.items():
        bar = "█" * int(median / 50)
        print(f"  {name:<12}: {median:6.0f}ms  {bar}")

    # ── Overall distribution summary ──
    print(f"\nDistribution Summary (ms):")
    print(f"  Real  — median: {np.median(real_arr):.1f}  p25: {np.percentile(real_arr,25):.1f}  p75: {np.percentile(real_arr,75):.1f}  p95: {np.percentile(real_arr,95):.1f}")
    print(f"  GAN   — median: {np.median(gen_arr):.1f}  p25: {np.percentile(gen_arr,25):.1f}  p75: {np.percentile(gen_arr,75):.1f}  p95: {np.percentile(gen_arr,95):.1f}")
    print(f"  Samples — Real: {len(real_arr):,}  GAN: {len(gen_arr):,}")

    # ── Fair KS test: subsample real to same size as generated ──
    rng = np.random.default_rng(42)
    real_sub = rng.choice(real_arr, size=min(len(gen_arr), len(real_arr)), replace=False)
    ks_stat, p_value = stats.ks_2samp(real_sub, gen_arr)

    print(f"\nKolmogorov-Smirnov Test (subsampled real n={len(real_sub):,}):")
    print(f"  KS Statistic: {ks_stat:.4f}  (target < 0.10)")
    print(f"  P-value:      {p_value:.4f}")
    if ks_stat < 0.10:
        print("  Result: EXCELLENT — distributions nearly identical")
    elif ks_stat < 0.20:
        print("  Result: GOOD — minor differences, acceptable for production")
    elif ks_stat < 0.30:
        print("  Result: FAIR — noticeable but usable; consider more epochs")
    else:
        print(f"  Result: POOR — significant gap; recommend more training (current: {ks_stat:.2f})")


if __name__ == "__main__":
    main()
