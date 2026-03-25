#!/usr/bin/env python3
"""Standalone HMM training script."""
import json
import pickle
import numpy as np
import click
from pathlib import Path
from loguru import logger

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


@click.command()
@click.option("--data", default="data/raw/keystroke_samples.jsonl")
@click.option("--output-dir", default="models")
@click.option("--n-states", default=6)
@click.option("--n-iter", default=100)
def main(data, output_dir, n_states, n_iter):
    if not HMM_AVAILABLE:
        logger.error("hmmlearn not available")
        return

    # Load data
    sequences = []
    path = Path(data)
    if path.exists():
        with open(path) as f:
            for line in f:
                record = json.loads(line.strip())
                timings = record.get("timings", [])
                if timings:
                    delays = [t[0] if isinstance(t, list) else t for t in timings]
                    sequences.append(np.array(delays, dtype=np.float32).reshape(-1, 1))
    else:
        logger.warning(f"No data at {data}, generating synthetic sequences")
        for _ in range(200):
            seq = np.random.lognormal(np.log(120), 0.5, (32, 1)).astype(np.float32)
            sequences.append(seq)

    X = np.concatenate(sequences, axis=0)
    lengths = [len(s) for s in sequences]

    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=n_iter)
    model.fit(X, lengths)

    Path(output_dir).mkdir(exist_ok=True)
    with open(f"{output_dir}/hmm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info(f"HMM model saved to {output_dir}/hmm_model.pkl")


if __name__ == "__main__":
    main()
