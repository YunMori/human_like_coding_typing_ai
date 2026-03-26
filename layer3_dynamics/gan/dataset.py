import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from loguru import logger


class KeystrokeDataset(Dataset):
    """JSONL dataset where each line is a keystroke sequence.

    Each record may include:
        timings:   [[delay_ms, hold_ms, gap_ms], ...]  32 steps
        context:   [[ctx_dim, ...], ...]               32 × 32 pre-computed context vectors
    """

    def __init__(self, jsonl_path: str, seq_len: int = 32):
        self.seq_len = seq_len
        self.sequences = []   # list of {"timings": np, "context": np}
        path = Path(jsonl_path)
        if path.exists():
            self._load(path)
        else:
            logger.warning(f"Dataset not found at {jsonl_path}, using synthetic data")
            self._generate_synthetic(1000)

    def _load(self, path: Path):
        with open(path) as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    timings = record.get("timings", [])
                    if len(timings) < self.seq_len:
                        continue

                    timing_arr = np.array(timings[:self.seq_len], dtype=np.float32)
                    timing_arr = np.clip(timing_arr / 1000.0, 0, 5)  # ms → s, clip @ 5s

                    # Load pre-computed context if present, else zero
                    context = record.get("context", None)
                    if context and len(context) >= self.seq_len:
                        ctx_arr = np.array(context[:self.seq_len], dtype=np.float32)
                    else:
                        ctx_arr = np.zeros((self.seq_len, 32), dtype=np.float32)

                    self.sequences.append({"timings": timing_arr, "context": ctx_arr})
                except Exception:
                    continue
        logger.info(f"Loaded {len(self.sequences)} keystroke sequences")

    def _generate_synthetic(self, n: int):
        for _ in range(n):
            timing_arr = np.random.lognormal(np.log(0.12), 0.4, (self.seq_len, 3)).astype(np.float32)
            timing_arr = np.clip(timing_arr, 0.02, 5.0)
            ctx_arr = np.zeros((self.seq_len, 32), dtype=np.float32)
            self.sequences.append({"timings": timing_arr, "context": ctx_arr})

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        record = self.sequences[idx]
        return {
            "timings": torch.FloatTensor(record["timings"]),   # (seq_len, 3)
            "context": torch.FloatTensor(record["context"]),   # (seq_len, 32)
        }
