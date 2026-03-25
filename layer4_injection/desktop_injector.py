import asyncio
import time
from typing import List
from loguru import logger

from layer4_injection.base_injector import BaseInjector
from layer3_dynamics.keystroke_event import KeystrokeEvent, EventType
from core.exceptions import InjectionError

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logger.warning("pyautogui not available")


class DesktopInjector(BaseInjector):
    def __init__(self, config: dict):
        super().__init__(config)
        self.dry_run = config.get("dry_run", False)
        self._typed_keys = []  # for dry run / testing

    async def setup(self, **kwargs):
        if not PYAUTOGUI_AVAILABLE and not self.dry_run:
            raise InjectionError("pyautogui is required for desktop injection")
        logger.info("Desktop injector ready")

    async def inject(self, events: List[KeystrokeEvent]):
        for event in events:
            if event.delay_before_ms > 0:
                await asyncio.sleep(event.delay_before_ms / 1000.0)

            if event.event_type == EventType.PAUSE:
                continue

            if self.dry_run:
                self._typed_keys.append(event.key)
                continue

            if not PYAUTOGUI_AVAILABLE:
                continue

            try:
                if event.event_type == EventType.BACKSPACE:
                    pyautogui.press("backspace")
                elif event.key:
                    pyautogui.keyDown(event.key)
                    await asyncio.sleep(event.key_hold_ms / 1000.0)
                    pyautogui.keyUp(event.key)
            except Exception as e:
                logger.warning(f"Injection error for key '{event.key}': {e}")

    async def teardown(self):
        pass

    def get_typed_text(self) -> str:
        return "".join(k for k in self._typed_keys if k and len(k) == 1)
