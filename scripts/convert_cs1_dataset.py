#!/usr/bin/env python3
"""
Convert 2019 CS1 Keystroke Dataset to JSONL format for GAN/HMM training.

Input:  keystrokes.csv  (ProgSnap2 format, ClientTimestamp in ms)
Output: data/raw/keystroke_samples.jsonl
        Each line: {"timings": [[delay_ms, hold_ms, gap_ms], ...], "subject": "S000"}
"""
import json
import click
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger


SEQ_LEN = 32
MAX_PAUSE_MS = 5000   # gaps longer than 5s = new session boundary
MIN_DELAY_MS = 10     # below this = likely duplicate event, skip


def extract_sequences(df: pd.DataFrame, seq_len: int) -> list[dict]:
    sequences = []

    # Process per subject, sorted by EventID (handles same-timestamp ordering)
    for subject_id, group in df.groupby("SubjectID"):
        group = group.sort_values(["ClientTimestamp", "EventID"])
        timestamps = group["ClientTimestamp"].values  # ms

        # Compute inter-keystroke intervals
        delays = np.diff(timestamps).astype(float)  # ms

        # Split into sessions at long pauses
        session_breaks = np.where(delays > MAX_PAUSE_MS)[0] + 1
        sessions = np.split(np.arange(len(timestamps)), session_breaks)

        for session_idx in sessions:
            if len(session_idx) < seq_len + 1:
                continue

            session_delays = delays[session_idx[:-1]]

            # Filter out suspiciously small delays (deduplication artifacts)
            session_delays = np.where(session_delays < MIN_DELAY_MS, MIN_DELAY_MS, session_delays)

            # Slide window over session with stride = seq_len
            for start in range(0, len(session_delays) - seq_len + 1, seq_len):
                window = session_delays[start:start + seq_len]
                if len(window) < seq_len:
                    continue

                # Estimate [keydown_ms, keyhold_ms, gap_ms]
                # keyhold_ms: ~30% of delay (heuristic, no keyup events in data)
                # gap_ms: ~70% of delay (time between keyup and next keydown)
                hold = window * 0.3
                gap = window * 0.7

                timings = np.stack([window, hold, gap], axis=1).tolist()
                sequences.append({"timings": timings, "subject": subject_id})

    return sequences


@click.command()
@click.argument("csv_path", default="/Users/yunjongseo/Downloads/2019 CS1 Keystroke Data/keystrokes.csv")
@click.option("--output", default="data/raw/keystroke_samples.jsonl", show_default=True)
@click.option("--seq-len", default=SEQ_LEN, show_default=True)
@click.option("--sample", default=0, help="Use only first N rows (0 = all). Useful for testing.")
def main(csv_path, output, seq_len, sample):
    """Convert CS1 keystroke CSV to JSONL training data."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        return

    logger.info(f"Loading {csv_path} ...")
    usecols = ["EventID", "SubjectID", "ClientTimestamp", "X-Keystroke"]
    df = pd.read_csv(csv_path, usecols=usecols, nrows=sample if sample else None)
    logger.info(f"Loaded {len(df):,} rows")

    # Keep only actual keystroke events
    df = df.dropna(subset=["X-Keystroke", "ClientTimestamp"])
    df["ClientTimestamp"] = df["ClientTimestamp"].astype(np.int64)
    df["EventID"] = df["EventID"].astype(np.int64)
    logger.info(f"Keystroke events: {len(df):,}")

    logger.info("Extracting sequences ...")
    sequences = extract_sequences(df, seq_len)
    logger.info(f"Extracted {len(sequences):,} sequences from {df['SubjectID'].nunique()} subjects")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for seq in sequences:
            f.write(json.dumps(seq) + "\n")

    logger.success(f"Saved to {output_path}  ({len(sequences):,} sequences)")

    # Quick stats
    all_delays = []
    for seq in sequences[:500]:
        all_delays.extend([t[0] for t in seq["timings"]])
    arr = np.array(all_delays)
    logger.info(
        f"Delay stats (ms) — median: {np.median(arr):.0f}  "
        f"p25: {np.percentile(arr, 25):.0f}  "
        f"p75: {np.percentile(arr, 75):.0f}  "
        f"p95: {np.percentile(arr, 95):.0f}"
    )


if __name__ == "__main__":
    main()
