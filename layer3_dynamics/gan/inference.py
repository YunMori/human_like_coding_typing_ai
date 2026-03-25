import torch
import numpy as np
from typing import Optional, List
from pathlib import Path
from loguru import logger

from layer3_dynamics.gan.generator import TimingGenerator


class GANInference:
    def __init__(self, model_path: Optional[str] = None, config: dict = None):
        config = config or {}
        self.noise_dim = config.get("noise_dim", 64)
        self.context_dim = config.get("context_dim", 32)
        self.seq_len = config.get("seq_len", 32)
        self.device = torch.device("cpu")

        self.G = TimingGenerator(
            noise_dim=self.noise_dim,
            context_dim=self.context_dim,
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 3),
            seq_len=self.seq_len,
        )

        self.trained = False
        if model_path and Path(model_path).exists():
            self.G.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.trained = True
            logger.info(f"Loaded GAN generator from {model_path}")
        else:
            logger.info("No GAN model found, using untrained generator (fallback to HMM)")

        self.G.eval()

    def sample_timings(self, context_vector: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Returns (n_samples, seq_len, 3) array of [key_down_ms, key_up_ms, gap_ms].
        Values are in milliseconds.
        """
        with torch.no_grad():
            noise = torch.randn(n_samples, self.noise_dim)
            ctx = torch.FloatTensor(context_vector).unsqueeze(0).expand(n_samples, -1)
            timings = self.G(noise, ctx)  # (n, seq_len, 3)
            # Convert from normalized seconds back to ms
            timings_ms = timings.numpy() * 1000.0
        return timings_ms

    def build_context_vector(self, complexity: int, fatigue: float,
                              hmm_state: int, prev_key: str = " ",
                              is_bigram: bool = False) -> np.ndarray:
        """Build 32-dim context vector."""
        ctx = np.zeros(self.context_dim, dtype=np.float32)
        ctx[0] = complexity / 4.0        # normalized complexity
        ctx[1] = fatigue                  # fatigue ratio
        # HMM state one-hot (6 dims, indices 2-7)
        if 0 <= hmm_state < 6:
            ctx[2 + hmm_state] = 1.0
        # prev_key embedding (23 dims, indices 8-30): simple char code normalized
        if prev_key:
            ctx[8] = ord(prev_key[0]) / 128.0
        ctx[31] = float(is_bigram)
        return ctx
