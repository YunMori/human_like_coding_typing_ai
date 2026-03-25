import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

DEFAULT_FAST_BIGRAMS = {
    "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
    "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
    "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
}


class BigramModel:
    def __init__(self, freq_path: Optional[str] = None):
        self.fast_bigrams = set(DEFAULT_FAST_BIGRAMS)
        self.bigram_speedup: Dict[str, float] = {}
        if freq_path and Path(freq_path).exists():
            self._load_frequencies(freq_path)
        self._build_speedup_table()

    def _load_frequencies(self, path: str):
        with open(path) as f:
            data = json.load(f)
        total = sum(data.values())
        top_n = sorted(data.items(), key=lambda x: -x[1])[:50]
        self.fast_bigrams = {bg for bg, _ in top_n}
        logger.info(f"Loaded {len(self.fast_bigrams)} fast bigrams from {path}")

    def _build_speedup_table(self):
        for bg in self.fast_bigrams:
            # Common bigrams get 10-30% speedup
            self.bigram_speedup[bg] = np.random.uniform(0.7, 0.9)

    def get_speedup(self, prev_char: str, curr_char: str) -> float:
        """Return timing multiplier (< 1.0 = faster) for this bigram."""
        bg = (prev_char + curr_char).lower()
        return self.bigram_speedup.get(bg, 1.0)
