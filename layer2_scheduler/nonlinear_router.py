import numpy as np
from typing import List
from layer2_scheduler.typing_plan import TypingSegment, ComplexityLevel
from loguru import logger


class NonlinearRouter:
    def __init__(self, config: dict):
        self.backvisit_prob = config.get("backvisit_probability", 0.15)

    def route(self, segments: List[TypingSegment]) -> List[TypingSegment]:
        """Occasionally insert back-visit segments (re-reading previous code)."""
        if len(segments) < 3:
            return segments
        result = []
        for i, seg in enumerate(segments):
            result.append(seg)
            # With some probability, insert a "back visit" (pause simulating re-reading)
            if i > 0 and np.random.random() < self.backvisit_prob:
                backvisit = TypingSegment(
                    text="",  # no typing, just pause
                    complexity=ComplexityLevel.MODERATE,
                    pause_before_ms=float(np.random.uniform(500, 3000)),
                    is_backvisit=True,
                )
                result.append(backvisit)
        return result
