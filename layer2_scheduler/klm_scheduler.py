from typing import List
from layer2_scheduler.typing_plan import ComplexityLevel, KLMOperator, TypingSegment
from loguru import logger

SPECIAL_CHARS = set("()[]{}\"'<>!@#$%^&*-+=|\\/:;,.")


class KLMScheduler:
    def __init__(self, config: dict):
        self.K = config.get("K", 0.28) * 1000  # ms
        self.M = config.get("M", 1.35) * 1000
        self.H = config.get("H", 0.40) * 1000
        self.P = config.get("P", 1.10) * 1000

    def schedule(self, segments: List[TypingSegment]) -> List[TypingSegment]:
        """Assign KLM operators and compute pause_before_ms for each segment."""
        for i, seg in enumerate(segments):
            ops = []
            # Mental operator for MODERATE and above
            if seg.complexity.value >= ComplexityLevel.MODERATE.value:
                ops.append(KLMOperator.M)
            # Keystroke operators
            ops.extend([KLMOperator.K] * max(1, len(seg.text) // 10))
            seg.klm_ops = ops
            # Compute pause
            pause = 0.0
            for op in ops:
                if op == KLMOperator.M:
                    pause += self.M
                elif op == KLMOperator.H:
                    pause += self.H
                elif op == KLMOperator.P:
                    pause += self.P
            # Extra pause for special chars at start
            if seg.text and seg.text[0] in SPECIAL_CHARS:
                pause += self._micro_pause()
            seg.pause_before_ms = max(seg.pause_before_ms, pause)
        return segments

    def _micro_pause(self) -> float:
        import numpy as np
        return float(np.random.uniform(50, 200))  # 50-200ms micro
