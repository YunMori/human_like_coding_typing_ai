import json
from layer4_injection.base_injector import BaseInjector


class JsonOutputInjector(BaseInjector):
    def __init__(self, config: dict):
        super().__init__(config)
        self.collected_events = []

    async def setup(self, **kwargs):
        pass

    async def inject(self, events):
        self.collected_events = list(events)

    async def teardown(self):
        pass

    def to_json(self) -> str:
        return json.dumps([
            {
                "key": e.key,
                "delay_before_ms": e.delay_before_ms,
                "key_hold_ms": e.key_hold_ms,
                "event_type": e.event_type.value,
                "is_error": e.is_error,
                "is_correction": e.is_correction,
            }
            for e in self.collected_events
        ])
