import numpy as np
from typing import List
from layer3_dynamics.keystroke_event import KeystrokeEvent, EventType
from loguru import logger


class CorrectionEngine:
    def generate_correction_events(
        self,
        error_text: str,
        intended_text: str,
        base_delay_ms: float = 100.0,
    ) -> List[KeystrokeEvent]:
        """
        Generate: 2-4 chars typed after error -> micro pause -> backspaces -> retype correct.
        """
        events = []
        # Type 2-4 more chars before noticing
        extra_chars = int(np.random.randint(2, 5))
        # Micro pause before correction
        pause_ms = float(np.random.uniform(500, 1500))
        # Add pause event
        events.append(KeystrokeEvent(
            key="",
            delay_before_ms=pause_ms,
            key_hold_ms=0,
            event_type=EventType.PAUSE,
            is_correction=True,
        ))
        # Backspace for error + extra chars
        n_backspaces = len(error_text) + extra_chars
        for i in range(n_backspaces):
            events.append(KeystrokeEvent(
                key="\x08",  # backspace
                delay_before_ms=float(np.random.uniform(60, 140)),
                key_hold_ms=float(np.random.uniform(30, 80)),
                event_type=EventType.BACKSPACE,
                is_correction=True,
            ))
        # Retype correct text
        for ch in intended_text:
            events.append(KeystrokeEvent(
                key=ch,
                delay_before_ms=float(np.random.uniform(80, 180)),
                key_hold_ms=float(np.random.uniform(40, 100)),
                event_type=EventType.KEYDOWN,
                is_correction=True,
            ))
        return events
