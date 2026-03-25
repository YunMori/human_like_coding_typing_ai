from abc import ABC, abstractmethod
from typing import List
from layer3_dynamics.keystroke_event import KeystrokeEvent


class BaseInjector(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    async def setup(self, **kwargs):
        """Initialize the injection target."""
        pass

    @abstractmethod
    async def inject(self, events: List[KeystrokeEvent]):
        """Inject keystroke events."""
        pass

    @abstractmethod
    async def teardown(self):
        """Clean up resources."""
        pass
