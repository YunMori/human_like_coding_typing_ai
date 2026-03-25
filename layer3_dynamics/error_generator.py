import numpy as np
from typing import Optional, List, Tuple
from layer3_dynamics.keyboard_layout import get_neighbor_keys
from loguru import logger


ErrorType = str  # "neighbor" | "swap" | "double" | "omit"


class ErrorGenerator:
    def __init__(self, config: dict):
        self.base_error_rate = config.get("error_rate", 0.02)

    def should_make_error(self, hmm_state: int) -> bool:
        state_error_rates = {
            0: self.base_error_rate,       # NORMAL
            1: self.base_error_rate * 0.5,  # SLOW
            2: self.base_error_rate * 2.0,  # FAST
            3: 0.15,                        # ERROR state
            4: 0.0,                         # CORRECTION
            5: 0.0,                         # PAUSE
        }
        rate = state_error_rates.get(hmm_state, self.base_error_rate)
        return np.random.random() < rate

    def generate_error(self, intended_char: str) -> Tuple[str, ErrorType]:
        """Return (actual_typed_char, error_type)."""
        error_type = np.random.choice(
            ["neighbor", "swap", "double", "omit"],
            p=[0.5, 0.2, 0.2, 0.1],
        )
        if error_type == "neighbor":
            neighbors = get_neighbor_keys(intended_char)
            if neighbors:
                return np.random.choice(neighbors), "neighbor"
        elif error_type == "double":
            return intended_char + intended_char, "double"
        elif error_type == "omit":
            return "", "omit"
        # swap - return intended but flag for later swap
        return intended_char, "swap"

    def get_error_chars(self, intended: str, n_errors: int = 1) -> List[Tuple[str, ErrorType]]:
        results = []
        for _ in range(n_errors):
            results.append(self.generate_error(intended))
        return results
