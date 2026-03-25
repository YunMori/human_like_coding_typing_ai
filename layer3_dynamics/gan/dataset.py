import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
from loguru import logger


class KeystrokeDataset(Dataset):
    """JSONL dataset where each line is a keystroke sequence."""

    def __init__(self, jsonl_path: str, seq_len: int = 32):
        self.seq_len = seq_len
        self.sequences = []
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
                    if len(timings) >= self.seq_len:
                        seq = np.array(timings[:self.seq_len], dtype=np.float32)
                        # Normalize
                        seq = np.clip(seq / 1000.0, 0, 5)  # convert ms to seconds, clip at 5s
                        self.sequences.append(seq)
                except Exception:
                    continue
        logger.info(f"Loaded {len(self.sequences)} keystroke sequences")

    def _generate_synthetic(self, n: int):
        """Generate synthetic human-like timing data for bootstrapping."""
        for _ in range(n):
            # Simulate inter-key intervals: mostly 80-200ms with occasional pauses
            seq = np.random.lognormal(np.log(0.12), 0.4, (self.seq_len, 3)).astype(np.float32)
            seq = np.clip(seq, 0.02, 5.0)
            self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx])
