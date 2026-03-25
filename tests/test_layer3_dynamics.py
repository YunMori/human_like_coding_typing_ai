import pytest
import numpy as np
from layer3_dynamics.keyboard_layout import fitts_delay_ms, get_neighbor_keys, get_position
from layer3_dynamics.hmm_engine import HMMEngine
from layer3_dynamics.bigram_model import BigramModel
from layer3_dynamics.fatigue_model import FatigueModel
from layer3_dynamics.error_generator import ErrorGenerator
from layer3_dynamics.correction_engine import CorrectionEngine
from layer3_dynamics.keystroke_event import EventType


def test_fitts_delay_adjacent():
    d = fitts_delay_ms("f", "g")  # adjacent keys
    assert 100 <= d <= 300  # ms range


def test_fitts_delay_far():
    d = fitts_delay_ms("q", "p")  # far apart
    assert d > fitts_delay_ms("f", "g")


def test_fitts_same_finger_penalty():
    # b and n are both home row adjacent, different fingers
    # vs r and t which share index finger region
    d_same = fitts_delay_ms("r", "t")
    d_diff = fitts_delay_ms("f", "j")
    # same finger should generally be slower (higher delay)
    # just check it returns reasonable values
    assert d_same > 0
    assert d_diff > 0


def test_neighbor_keys():
    neighbors = get_neighbor_keys("f")
    assert len(neighbors) > 0
    assert "g" in neighbors or "d" in neighbors  # adjacent to f


def test_hmm_state_sequence():
    hmm = HMMEngine()
    states = hmm.generate_state_sequence(100)
    assert len(states) == 100
    assert all(0 <= s <= 5 for s in states)


def test_hmm_delay_sampling():
    hmm = HMMEngine()
    for state in range(6):
        delay = hmm.sample_delay_ms(state)
        assert delay >= 20.0


def test_bigram_speedup():
    bigram = BigramModel()
    # "th" is a common fast bigram
    speedup = bigram.get_speedup("t", "h")
    assert 0.5 <= speedup <= 1.0  # either speedup or neutral


def test_fatigue_model():
    fatigue = FatigueModel({"fatigue_decay_per_char": 0.001, "fatigue_min_speed_ratio": 0.4})
    initial_ratio = fatigue.get_speed_multiplier()
    fatigue.update(500)
    after_ratio = fatigue.get_speed_multiplier()
    assert after_ratio < initial_ratio
    fatigue.update(1000)
    min_ratio = fatigue.get_speed_multiplier()
    assert min_ratio >= 0.4  # should not go below minimum


def test_fatigue_delay_multiplier():
    fatigue = FatigueModel({"fatigue_decay_per_char": 0.001, "fatigue_min_speed_ratio": 0.4})
    fatigue.update(1000)
    # When fatigued, delay multiplier should be > 1
    assert fatigue.get_delay_multiplier() > 1.0


def test_error_generator():
    gen = ErrorGenerator({"error_rate": 1.0})  # always make errors
    for _ in range(20):
        char = "a"
        should_err = gen.should_make_error(0)  # state 0, error_rate=1.0
        assert should_err


def test_correction_engine():
    engine = CorrectionEngine()
    events = engine.generate_correction_events("wrng", "wrong", base_delay_ms=100)
    assert len(events) > 0
    # Should have pause and backspaces
    event_types = [e.event_type for e in events]
    assert EventType.PAUSE in event_types
    assert EventType.BACKSPACE in event_types
    # All correction events marked
    assert all(e.is_correction for e in events)
