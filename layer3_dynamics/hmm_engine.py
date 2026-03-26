import numpy as np
from typing import List, Optional
from pathlib import Path
from loguru import logger

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available, using fallback HMM")

HMM_STATE_NAMES = ["NORMAL", "SLOW", "FAST", "ERROR", "CORRECTION", "PAUSE"]
N_STATES = 6

# Default transition matrix (row=from, col=to)
DEFAULT_TRANSMAT = np.array([
    [0.80, 0.07, 0.07, 0.03, 0.02, 0.01],  # NORMAL
    [0.30, 0.55, 0.05, 0.05, 0.03, 0.02],  # SLOW
    [0.30, 0.05, 0.55, 0.07, 0.02, 0.01],  # FAST
    [0.10, 0.10, 0.05, 0.30, 0.40, 0.05],  # ERROR
    [0.50, 0.15, 0.10, 0.05, 0.15, 0.05],  # CORRECTION
    [0.40, 0.20, 0.10, 0.05, 0.05, 0.20],  # PAUSE
])

DEFAULT_STARTPROB = np.array([0.6, 0.15, 0.10, 0.05, 0.05, 0.05])

# Mean/std of delay_ms for each state
STATE_DELAY_PARAMS = {
    0: (120, 30),    # NORMAL
    1: (250, 60),    # SLOW
    2: (60, 15),     # FAST
    3: (180, 80),    # ERROR
    4: (100, 40),    # CORRECTION
    5: (2000, 1000), # PAUSE
}


class HMMEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.transmat = DEFAULT_TRANSMAT
        self.startprob = DEFAULT_STARTPROB
        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _load_model(self, path: str):
        import pickle
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded HMM model from {path}")

    def generate_state_sequence(self, length: int, initial_state: int = 0) -> List[int]:
        """Generate HMM state sequence of given length."""
        states = [initial_state]
        for _ in range(length - 1):
            current = states[-1]
            probs = self.transmat[current]
            next_state = int(np.random.choice(N_STATES, p=probs))
            states.append(next_state)
        return states

    def sample_delay_ms(self, state: int) -> float:
        """Sample keystroke delay for a given HMM state."""
        mean, std = STATE_DELAY_PARAMS[state]
        delay = float(np.random.normal(mean, std))
        return max(20.0, delay)  # minimum 20ms

    def sample_hold_ms(self, state: int) -> float:
        """Sample key hold duration for a given HMM state."""
        base_hold = {0: 60, 1: 90, 2: 40, 3: 70, 4: 65, 5: 55}
        mean = base_hold.get(state, 60)
        return max(20.0, float(np.random.normal(mean, mean * 0.2)))

    def decode_sequence(self, delays_ms: np.ndarray) -> List[int]:
        """Viterbi decoding: IKI array (ms) → state index sequence.
        Uses trained HMM model if loaded, otherwise falls back to heuristic.
        """
        if self.model is None:
            return [self._heuristic_state(float(d)) for d in delays_ms]
        obs = np.array(delays_ms, dtype=np.float32).reshape(-1, 1)
        states = self.model.predict(obs)
        return states.tolist()

    @staticmethod
    def _heuristic_state(delay_ms: float) -> int:
        """Threshold-based fallback (used only when HMM model not loaded)."""
        if delay_ms >= 1000:
            return 5
        if delay_ms >= 350:
            return 1
        if delay_ms >= 220:
            return 3
        if delay_ms >= 80:
            return 0
        if delay_ms >= 45:
            return 4
        return 2
