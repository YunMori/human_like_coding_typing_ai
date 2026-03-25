#!/usr/bin/env python3
"""Benchmark GAN quality using KS test."""
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
        # Generate synthetic reference
        real_delays = np.random.lognormal(np.log(120), 0.5, 1000).tolist()
        logger.warning("No real data found, using synthetic reference")

    # Generate GAN samples
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
    ctx = gan.build_context_vector(complexity=2, fatigue=1.0, hmm_state=0)
    gen_timings = gan.sample_timings(ctx, n_samples=32)  # (32, seq_len, 3)
    gen_delays = gen_timings[:, :, 0].flatten().tolist()  # key_down_ms

    ks_stat, p_value = stats.ks_2samp(real_delays, gen_delays)
    print(f"\nKolmogorov-Smirnov Test Results:")
    print(f"  KS Statistic: {ks_stat:.4f}")
    print(f"  P-value:      {p_value:.4f}")
    if p_value > 0.05:
        print("  Result: PASS - Generated timings are statistically indistinguishable from human (p > 0.05)")
    else:
        print(f"  Result: FAIL - Distributions differ significantly (p = {p_value:.4f})")
        print("  Consider training with more data or more epochs.")


if __name__ == "__main__":
    main()
