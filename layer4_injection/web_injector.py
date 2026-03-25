import asyncio
from typing import List, Optional
from loguru import logger

from layer4_injection.base_injector import BaseInjector
from layer3_dynamics.keystroke_event import KeystrokeEvent, EventType
from core.exceptions import InjectionError

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("playwright not available")


class WebInjector(BaseInjector):
    def __init__(self, config: dict):
        super().__init__(config)
        self.playwright = None
        self.browser = None
        self.page = None
        self.dry_run = config.get("dry_run", False)
        self._typed_keys = []

    async def setup(self, url: str = None, selector: str = None, **kwargs):
        self.url = url
        self.selector = selector
        if self.dry_run:
            logger.info("Web injector in dry-run mode")
            return
        if not PLAYWRIGHT_AVAILABLE:
            raise InjectionError("playwright is required for web injection")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        self.page = await self.browser.new_page()
        if url:
            await self.page.goto(url)
            logger.info(f"Navigated to {url}")
        if selector:
            await self.page.click(selector)
            logger.info(f"Focused on selector: {selector}")

    async def inject(self, events: List[KeystrokeEvent]):
        for event in events:
            if event.delay_before_ms > 0:
                await asyncio.sleep(event.delay_before_ms / 1000.0)

            if event.event_type == EventType.PAUSE:
                continue

            if self.dry_run:
                self._typed_keys.append(event.key)
                continue

            if not self.page:
                continue

            try:
                if event.event_type == EventType.BACKSPACE:
                    await self.page.keyboard.press("Backspace")
                elif event.key:
                    await self.page.keyboard.down(event.key)
                    await asyncio.sleep(event.key_hold_ms / 1000.0)
                    await self.page.keyboard.up(event.key)
            except Exception as e:
                logger.warning(f"Web injection error for key '{event.key}': {e}")

    async def teardown(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    def get_typed_text(self) -> str:
        return "".join(k for k in self._typed_keys if k and len(k) == 1)
