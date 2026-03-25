from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EventType(Enum):
    KEYDOWN = "keydown"
    KEYUP = "keyup"
    BACKSPACE = "backspace"
    PAUSE = "pause"


@dataclass
class KeystrokeEvent:
    key: str
    delay_before_ms: float   # Time to wait BEFORE pressing this key
    key_hold_ms: float        # How long to hold the key
    event_type: EventType = EventType.KEYDOWN
    hmm_state: int = 0        # Which HMM state generated this
    is_error: bool = False
    is_correction: bool = False
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
