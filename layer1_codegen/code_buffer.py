from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DependencyInfo:
    imports: list = field(default_factory=list)
    functions: list = field(default_factory=list)
    classes: list = field(default_factory=list)
    call_graph: dict = field(default_factory=dict)


@dataclass
class CodeBuffer:
    raw_code: str
    language: str
    dependency_info: Optional[DependencyInfo] = None
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.raw_code)

    def lines(self):
        return self.raw_code.splitlines()
