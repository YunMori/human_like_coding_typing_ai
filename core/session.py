from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class SessionConfig:
    prompt: str
    language: str
    target: str  # "desktop" | "web"
    url: Optional[str] = None
    selector: Optional[str] = None
    dry_run: bool = False


@dataclass
class IntegrationConfig:
    """Swift vvs 앱으로부터 받는 요청 포맷"""
    code: str
    language: str = "python"
    model_dir: str = "models"
    config_path: str = "config.yaml"
    seed: Optional[int] = None


@dataclass
class SessionResult:
    session_id: str
    prompt: str
    language: str
    generated_code: str
    total_keystrokes: int
    total_duration_ms: float
    error_count: int
    correction_count: int
    error_rate: float
    avg_wpm: float
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finalize(self):
        self.finished_at = datetime.now()
        return self
