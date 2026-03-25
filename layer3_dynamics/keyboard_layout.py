import numpy as np
from typing import Dict, Tuple

# QWERTY key positions (row, col) approximate
QWERTY_POSITIONS: Dict[str, Tuple[float, float]] = {
    # Row 0 (number row)
    '`': (0, 0), '1': (0, 1), '2': (0, 2), '3': (0, 3), '4': (0, 4),
    '5': (0, 5), '6': (0, 6), '7': (0, 7), '8': (0, 8), '9': (0, 9),
    '0': (0, 10), '-': (0, 11), '=': (0, 12),
    # Row 1 (QWERTY)
    'q': (1, 0.5), 'w': (1, 1.5), 'e': (1, 2.5), 'r': (1, 3.5), 't': (1, 4.5),
    'y': (1, 5.5), 'u': (1, 6.5), 'i': (1, 7.5), 'o': (1, 8.5), 'p': (1, 9.5),
    '[': (1, 10.5), ']': (1, 11.5), '\\': (1, 12.5),
    # Row 2 (ASDF)
    'a': (2, 0.75), 's': (2, 1.75), 'd': (2, 2.75), 'f': (2, 3.75), 'g': (2, 4.75),
    'h': (2, 5.75), 'j': (2, 6.75), 'k': (2, 7.75), 'l': (2, 8.75),
    ';': (2, 9.75), "'": (2, 10.75),
    # Row 3 (ZXCV)
    'z': (3, 1.25), 'x': (3, 2.25), 'c': (3, 3.25), 'v': (3, 4.25), 'b': (3, 5.25),
    'n': (3, 6.25), 'm': (3, 7.25), ',': (3, 8.25), '.': (3, 9.25), '/': (3, 10.25),
    # Space
    ' ': (4, 6.0),
}

# Finger assignments for QWERTY (0=left pinky, 1=left ring, ..., 7=right pinky)
FINGER_MAP: Dict[str, int] = {
    '`': 0, '1': 0, 'q': 0, 'a': 0, 'z': 0,
    '2': 1, 'w': 1, 's': 1, 'x': 1,
    '3': 2, 'e': 2, 'd': 2, 'c': 2,
    '4': 3, '5': 3, 'r': 3, 't': 3, 'f': 3, 'g': 3, 'v': 3, 'b': 3,
    '6': 4, '7': 4, 'y': 4, 'u': 4, 'h': 4, 'j': 4, 'n': 4, 'm': 4,
    '8': 5, 'i': 5, 'k': 5, ',': 5,
    '9': 6, 'o': 6, 'l': 6, '.': 6,
    '0': 7, '-': 7, '=': 7, 'p': 7, '[': 7, ']': 7, '\\': 7, ';': 7, "'": 7, '/': 7,
    ' ': 4,  # thumb
}

KEY_WIDTH = 1.0  # normalized key width


def get_position(key: str) -> Tuple[float, float]:
    k = key.lower()
    return QWERTY_POSITIONS.get(k, (2.0, 6.0))  # default to center


def fitts_delay_ms(key1: str, key2: str) -> float:
    """MT = 0.05 + 0.10 * log2(D/W + 1) seconds"""
    p1 = np.array(get_position(key1))
    p2 = np.array(get_position(key2))
    D = float(np.linalg.norm(p2 - p1))
    W = KEY_WIDTH
    mt_seconds = 0.05 + 0.10 * np.log2(D / W + 1)
    # Same finger penalty
    f1 = FINGER_MAP.get(key1.lower(), -1)
    f2 = FINGER_MAP.get(key2.lower(), -1)
    if f1 == f2 and f1 != -1:
        mt_seconds += 0.05
    return mt_seconds * 1000.0  # convert to ms


def get_neighbor_keys(key: str) -> list:
    """Return physically adjacent keys for typo generation."""
    pos = get_position(key)
    neighbors = []
    for k, p in QWERTY_POSITIONS.items():
        dist = np.sqrt((p[0] - pos[0])**2 + (p[1] - pos[1])**2)
        if 0 < dist <= 1.5:
            neighbors.append(k)
    return neighbors
