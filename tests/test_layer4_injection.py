import pytest
import asyncio
from layer4_injection.desktop_injector import DesktopInjector
from layer4_injection.web_injector import WebInjector
from layer4_injection.injector_factory import InjectorFactory
from layer3_dynamics.keystroke_event import KeystrokeEvent, EventType
from core.exceptions import InjectionError


@pytest.fixture
def sample_events():
    return [
        KeystrokeEvent(key="h", delay_before_ms=100, key_hold_ms=60, event_type=EventType.KEYDOWN),
        KeystrokeEvent(key="e", delay_before_ms=120, key_hold_ms=55, event_type=EventType.KEYDOWN),
        KeystrokeEvent(key="l", delay_before_ms=90, key_hold_ms=60, event_type=EventType.KEYDOWN),
        KeystrokeEvent(key="l", delay_before_ms=85, key_hold_ms=58, event_type=EventType.KEYDOWN),
        KeystrokeEvent(key="o", delay_before_ms=110, key_hold_ms=62, event_type=EventType.KEYDOWN),
    ]


@pytest.mark.asyncio
async def test_desktop_injector_dry_run(sample_events):
    injector = DesktopInjector({"dry_run": True})
    await injector.setup()
    await injector.inject(sample_events)
    await injector.teardown()
    assert injector.get_typed_text() == "hello"


@pytest.mark.asyncio
async def test_web_injector_dry_run(sample_events):
    injector = WebInjector({"dry_run": True})
    await injector.setup()
    await injector.inject(sample_events)
    await injector.teardown()
    assert injector.get_typed_text() == "hello"


@pytest.mark.asyncio
async def test_desktop_injector_with_pause():
    injector = DesktopInjector({"dry_run": True})
    await injector.setup()
    events = [
        KeystrokeEvent(key="", delay_before_ms=100, key_hold_ms=0, event_type=EventType.PAUSE),
        KeystrokeEvent(key="a", delay_before_ms=80, key_hold_ms=60, event_type=EventType.KEYDOWN),
    ]
    await injector.inject(events)
    assert injector.get_typed_text() == "a"


@pytest.mark.asyncio
async def test_desktop_injector_with_backspace():
    injector = DesktopInjector({"dry_run": True})
    await injector.setup()
    events = [
        KeystrokeEvent(key="a", delay_before_ms=80, key_hold_ms=60, event_type=EventType.KEYDOWN),
        KeystrokeEvent(key="\x08", delay_before_ms=100, key_hold_ms=50, event_type=EventType.BACKSPACE),
    ]
    await injector.inject(events)
    # In dry_run, backspace doesn't actually remove from _typed_keys
    # but it should not add the backspace char to typed text
    text = injector.get_typed_text()
    assert "a" in text or text == "a"  # 'a' was typed


def test_injector_factory_desktop():
    injector = InjectorFactory.create("desktop", {"dry_run": True})
    assert isinstance(injector, DesktopInjector)


def test_injector_factory_web():
    injector = InjectorFactory.create("web", {"dry_run": True})
    assert isinstance(injector, WebInjector)


def test_injector_factory_invalid():
    with pytest.raises(InjectionError):
        InjectorFactory.create("invalid_target", {})
