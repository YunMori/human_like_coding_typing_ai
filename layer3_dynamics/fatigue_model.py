import numpy as np
from loguru import logger


class FatigueModel:
    def __init__(self, config: dict):
        self.decay_per_char = config.get("fatigue_decay_per_char", 0.0005)
        self.min_speed_ratio = config.get("fatigue_min_speed_ratio", 0.4)
        self.char_count = 0
        self.current_ratio = 1.0

    def reset(self):
        self.char_count = 0
        self.current_ratio = 1.0

    def update(self, n_chars: int = 1):
        self.char_count += n_chars
        self.current_ratio = max(
            self.min_speed_ratio,
            1.0 - self.decay_per_char * self.char_count,
        )

    def get_speed_multiplier(self) -> float:
        return self.current_ratio

    def get_delay_multiplier(self) -> float:
        """Inverse: higher delay when more fatigued."""
        return 1.0 / max(0.1, self.current_ratio)
