import numpy as np
from layer2_scheduler.typing_plan import ComplexityLevel
from loguru import logger


class PauseInjector:
    def __init__(self, config: dict):
        self.cfg = config

    def get_pause_ms(self, complexity: ComplexityLevel, context: str = "") -> float:
        """Return pause duration in ms before typing this segment."""
        if complexity == ComplexityLevel.BOILERPLATE:
            return self._sample_micro()
        elif complexity == ComplexityLevel.SIMPLE:
            return self._sample_micro() if np.random.random() < 0.3 else 0.0
        elif complexity == ComplexityLevel.MODERATE:
            return self._sample_short()
        elif complexity == ComplexityLevel.COMPLEX:
            return self._sample_short() if np.random.random() < 0.6 else self._sample_mid()
        else:  # VERY_COMPLEX
            return self._sample_mid() if np.random.random() < 0.7 else self._sample_long()

    def get_special_char_pause_ms(self) -> float:
        """Micro pause after special chars like ()[]{}\"' """
        return self._sample_micro()

    def _sample_micro(self) -> float:
        mn = self.cfg.get("micro_min", 2) * 1000
        mx = self.cfg.get("micro_max", 15) * 1000
        return self._lognormal_sample(mn, mx)

    def _sample_short(self) -> float:
        mn = self.cfg.get("short_min", 15) * 1000
        mx = self.cfg.get("short_max", 180) * 1000
        return self._lognormal_sample(mn, mx)

    def _sample_mid(self) -> float:
        mn = self.cfg.get("mid_min", 180) * 1000
        mx = self.cfg.get("mid_max", 600) * 1000
        return self._lognormal_sample(mn, mx)

    def _sample_long(self) -> float:
        mn = self.cfg.get("long_min", 600) * 1000
        return self._lognormal_sample(mn, mn * 3)

    def _lognormal_sample(self, min_ms: float, max_ms: float) -> float:
        mu = np.log((min_ms + max_ms) / 2)
        sigma = 0.4
        sample = np.random.lognormal(mu, sigma)
        return float(np.clip(sample, min_ms, max_ms))
