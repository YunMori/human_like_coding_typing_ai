#!/usr/bin/env python3
"""
Convert 2019/2021 CS1 Keystroke Datasets to JSONL format for GAN training.

Supports multiple CSV files (2019 and 2021 datasets merged).

Input:  keystrokes.csv  (ProgSnap2 format, ClientTimestamp in ms)
Output: data/raw/keystroke_samples.jsonl
        Each line: {
            "timings":   [[delay_ms, hold_ms, gap_ms], ...],  # 32 steps
            "chars":     ["d", "e", "f", ...],                # 32 chars
            "locations": [0.12, 0.13, ...],                   # 32 normalized positions
            "subject":   "S000"
        }
"""
import json
import sys
import click
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from layer3_dynamics.hmm_engine import HMMEngine


SEQ_LEN = 32
MAX_PAUSE_MS = 5000
MIN_DELAY_MS = 10


def _char_type(ch: str) -> float:
    """0=alpha, 0.5=digit, 1=special"""
    if not ch:
        return 1.0
    c = ch[0]
    if c.isalpha():
        return 0.0
    if c.isdigit():
        return 0.5
    return 1.0


def _estimate_hmm_state(delay_ms: float) -> int:
    """Estimate HMM state index from keystroke delay.
    States: 0=NORMAL, 1=SLOW, 2=FAST, 3=ERROR, 4=CORRECTION, 5=PAUSE
    """
    if delay_ms >= 1000:
        return 5  # PAUSE
    if delay_ms >= 350:
        return 1  # SLOW
    if delay_ms >= 220:
        return 3  # ERROR
    if delay_ms >= 80:
        return 0  # NORMAL
    if delay_ms >= 45:
        return 4  # CORRECTION
    return 2      # FAST


def _estimate_complexity(delay_ms: float) -> float:
    """Estimate complexity (0-1) from delay."""
    if delay_ms < 100:
        return 0.25   # SIMPLE  (complexity=1/4)
    if delay_ms < 250:
        return 0.5    # MODERATE (complexity=2/4)
    return 1.0        # VERY_COMPLEX (complexity=4/4)


def _build_context(char: str, location: float, delay_ms: float = 150.0,
                   prev_char: str = " ", session_progress: float = 0.0,
                   hmm_state: int = -1) -> list[float]:
    """Build 32-dim context vector for one keystroke.

    Layout (unified with layer3_dynamics/gan/inference.py):
        [0]    source_location (0-1)
        [1]    char_type (0=alpha, 0.5=digit, 1=special)
        [2]    curr_char ASCII / 128
        [3]    is_delete
        [4]    complexity_approx (estimated from delay)
        [5]    fatigue_approx (1 - session_progress)
        [6:12] HMM state one-hot (6 states) — Viterbi-decoded from trained model
        [12]   prev_char ASCII / 128
        [13]   is_bigram (0 at training time — not available)
        [14-31] reserved (0)
    """
    ctx = [0.0] * 32
    ctx[0] = float(location)
    ctx[1] = _char_type(char)
    ctx[2] = ord(char[0]) / 128.0 if char else 0.0
    ctx[3] = 1.0 if char in ("\x08", "backspace", "BackSpace") else 0.0
    ctx[4] = _estimate_complexity(delay_ms)
    ctx[5] = max(0.0, 1.0 - float(session_progress))   # fresh=1.0, tired→0.0
    if hmm_state < 0:
        hmm_state = _estimate_hmm_state(delay_ms)       # fallback if not provided
    ctx[6 + hmm_state] = 1.0                            # one-hot in [6:12]
    ctx[12] = ord(prev_char[0]) / 128.0 if prev_char else 0.0
    # ctx[13] = is_bigram — leave 0; bigram data not available at conversion time
    return ctx


def load_csv(csv_path: Path, sample: int) -> pd.DataFrame:
    """Load a CS1 keystroke CSV, handling 2019 and 2021 column differences."""
    # Peek at header to detect dataset version
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    has_xkey = "X-Keystroke" in header

    base_cols = ["EventID", "SubjectID", "ClientTimestamp", "InsertText", "SourceLocation"]
    if has_xkey:
        base_cols.append("X-Keystroke")

    df = pd.read_csv(csv_path, usecols=base_cols, nrows=sample if sample else None,
                     low_memory=False)

    # Derive a unified 'char' column: InsertText → X-Keystroke → "?"
    if "X-Keystroke" in df.columns:
        df["char"] = df["InsertText"].fillna(df["X-Keystroke"]).fillna("?")
    else:
        df["char"] = df["InsertText"].fillna("?")

    # Keep only rows that have a timestamp and are actual edits
    df = df.dropna(subset=["ClientTimestamp"])
    df = df[df["char"] != "?"]  # drop rows with no character info

    df["ClientTimestamp"] = df["ClientTimestamp"].astype(np.int64)
    df["EventID"] = df["EventID"].astype(np.int64)
    df["SourceLocation"] = pd.to_numeric(df["SourceLocation"], errors="coerce").fillna(0.0)

    return df


def extract_sequences(df: pd.DataFrame, seq_len: int) -> list[dict]:
    sequences = []

    # Load trained HMM model for Viterbi state labeling
    hmm_model_path = Path(__file__).resolve().parent.parent / "models" / "hmm_model.pkl"
    hmm_engine = HMMEngine(model_path=str(hmm_model_path))
    if hmm_engine.model is not None:
        logger.info("HMM model loaded — using Viterbi decoding for state labels")
    else:
        logger.warning("HMM model not found — falling back to heuristic state labels")

    for subject_id, group in df.groupby("SubjectID"):
        group = group.sort_values(["ClientTimestamp", "EventID"])
        timestamps = group["ClientTimestamp"].values
        chars_all = group["char"].values
        locs_all = group["SourceLocation"].values

        delays = np.diff(timestamps).astype(float)

        # Normalize SourceLocation within this subject's session
        loc_max = locs_all.max() if locs_all.max() > 0 else 1.0
        locs_norm = locs_all / loc_max

        # Session boundaries at long pauses
        session_breaks = np.where(delays > MAX_PAUSE_MS)[0] + 1
        sessions = np.split(np.arange(len(timestamps)), session_breaks)

        session_total = len(timestamps)

        for session_idx in sessions:
            if len(session_idx) < seq_len + 1:
                continue

            session_delays = delays[session_idx[:-1]]
            session_chars = chars_all[session_idx[:-1]]
            session_locs = locs_norm[session_idx[:-1]]

            session_delays = np.where(session_delays < MIN_DELAY_MS, MIN_DELAY_MS, session_delays)

            # Viterbi decode entire session delays at once (more accurate than per-step)
            hmm_states = hmm_engine.decode_sequence(session_delays)

            for start in range(0, len(session_delays) - seq_len + 1, seq_len):
                window = session_delays[start:start + seq_len]
                if len(window) < seq_len:
                    continue

                hold = window * 0.3
                gap = window * 0.7
                timings = np.stack([window, hold, gap], axis=1).tolist()

                chars_w = [str(c) for c in session_chars[start:start + seq_len]]
                locs_w = session_locs[start:start + seq_len].tolist()

                # Pre-compute per-keystroke context vectors (32×32)
                # Use Viterbi-decoded HMM state labels (trained model's state numbering)
                context = []
                for k, (c, l, d) in enumerate(zip(chars_w, locs_w, window.tolist())):
                    prev_c = chars_w[k - 1] if k > 0 else " "
                    sess_prog = float(session_idx[start + k]) / max(session_total, 1)
                    state = hmm_states[start + k]
                    context.append(_build_context(c, l, delay_ms=d,
                                                   prev_char=prev_c,
                                                   session_progress=sess_prog,
                                                   hmm_state=state))

                sequences.append({
                    "timings": timings,
                    "chars": chars_w,
                    "locations": locs_w,
                    "context": context,
                    "subject": subject_id,
                })

    return sequences


@click.command()
@click.argument("csv_paths", nargs=-1, required=True)
@click.option("--output", default="data/raw/keystroke_samples.jsonl", show_default=True)
@click.option("--seq-len", default=SEQ_LEN, show_default=True)
@click.option("--sample", default=0, help="Rows per CSV to load (0 = all).")
def main(csv_paths, output, seq_len, sample):
    """Convert one or more CS1 keystroke CSVs to JSONL training data.

    Supports both 2019 and 2021 dataset formats automatically.

    \b
    Example:
        python scripts/convert_cs1_dataset.py \\
            "/path/2019 CS1 Keystroke Data/keystrokes.csv" \\
            "/path/2021 CS1 Keystroke Data/keystrokes.csv"
    """
    all_sequences = []

    for csv_path_str in csv_paths:
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            logger.error(f"File not found: {csv_path}")
            continue

        logger.info(f"Loading {csv_path.parent.name} ...")
        df = load_csv(csv_path, sample)
        logger.info(f"  → {len(df):,} keystroke events from {df['SubjectID'].nunique()} subjects")

        seqs = extract_sequences(df, seq_len)
        logger.info(f"  → {len(seqs):,} sequences extracted")
        all_sequences.extend(seqs)

    if not all_sequences:
        logger.error("No sequences extracted.")
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for seq in all_sequences:
            f.write(json.dumps(seq) + "\n")

    logger.success(f"Saved {len(all_sequences):,} sequences → {output_path}")

    # Quick stats
    all_delays = []
    for seq in all_sequences[:500]:
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
