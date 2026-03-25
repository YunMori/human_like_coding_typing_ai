#!/usr/bin/env python3
"""Collect real keystroke timing data for GAN training."""
import json
import time
import click
from pathlib import Path
from loguru import logger

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logger.error("pynput not available. Install with: pip install pynput")


@click.command()
@click.option("--output", default="data/raw/keystroke_samples.jsonl")
@click.option("--duration", default=300, help="Recording duration in seconds")
def main(output, duration):
    if not PYNPUT_AVAILABLE:
        logger.error("pynput required for data collection")
        return

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    sequences = []
    current_seq = []
    last_time = None

    def on_press(key):
        nonlocal last_time
        now = time.time()
        try:
            char = key.char
        except AttributeError:
            char = str(key)

        delay_ms = (now - last_time) * 1000 if last_time else 0
        last_time = now
        current_seq.append({"key": char, "delay_ms": delay_ms, "ts": now})

        # Save sequence every 32 keystrokes
        if len(current_seq) >= 32:
            timings = [[k["delay_ms"], k["delay_ms"] * 0.5, k["delay_ms"] * 0.3]
                       for k in current_seq[:32]]
            sequences.append({"timings": timings})
            current_seq.clear()

    logger.info(f"Recording keystrokes for {duration}s. Press Ctrl+C to stop early.")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()

    with open(output, "w") as f:
        for seq in sequences:
            f.write(json.dumps(seq) + "\n")
    logger.info(f"Saved {len(sequences)} sequences to {output}")


if __name__ == "__main__":
    main()
