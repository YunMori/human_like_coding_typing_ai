from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ComplexityLevel(Enum):
    BOILERPLATE = 0
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    VERY_COMPLEX = 4


class KLMOperator(Enum):
    K = "K"  # keystroke
    M = "M"  # mental
    H = "H"  # hand movement
    P = "P"  # pointing


@dataclass
class TypingSegment:
    text: str
    complexity: ComplexityLevel
    klm_ops: List[KLMOperator] = field(default_factory=list)
    pause_before_ms: float = 0.0
    node_type: Optional[str] = None
    is_backvisit: bool = False


@dataclass
class TypingPlan:
    segments: List[TypingSegment]
    total_text: str
    estimated_duration_ms: float
    language: str
    metadata: dict = field(default_factory=dict)
