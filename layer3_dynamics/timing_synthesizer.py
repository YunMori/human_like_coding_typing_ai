import numpy as np
from typing import List, Optional
from loguru import logger

from layer2_scheduler.typing_plan import TypingPlan, TypingSegment, ComplexityLevel
from layer3_dynamics.keystroke_event import KeystrokeEvent, EventType
from layer3_dynamics.keyboard_layout import fitts_delay_ms
from layer3_dynamics.hmm_engine import HMMEngine
from layer3_dynamics.bigram_model import BigramModel
from layer3_dynamics.fatigue_model import FatigueModel
from layer3_dynamics.error_generator import ErrorGenerator
from layer3_dynamics.correction_engine import CorrectionEngine
from layer3_dynamics.gan.inference import GANInference

SPECIAL_CHARS = set("()[]{}\"'<>!@#$%^&*-+=|\\/:;,.")


class TimingSynthesizer:
    def __init__(self, config: dict, model_dir: str = "models"):
        self.config = config
        self.hmm = HMMEngine(f"{model_dir}/hmm_model.pkl")
        self.bigram = BigramModel("data/bigram_frequencies.json")
        self.fatigue = FatigueModel(config.get("typing", {}))
        self.error_gen = ErrorGenerator(config.get("typing", {}))
        self.correction = CorrectionEngine()
        self.gan = GANInference(
            model_path=f"{model_dir}/gan_generator.pth",
            config=config.get("gan", {}),
        )

    def synthesize(self, plan: TypingPlan) -> List[KeystrokeEvent]:
        self.fatigue.reset()
        events: List[KeystrokeEvent] = []
        all_text = plan.total_text
        char_count = 0

        for seg in plan.segments:
            # Back-visit: just add a pause
            if seg.is_backvisit:
                if seg.pause_before_ms > 0:
                    events.append(KeystrokeEvent(
                        key="", delay_before_ms=seg.pause_before_ms,
                        key_hold_ms=0, event_type=EventType.PAUSE,
                    ))
                continue

            # Segment-level pause
            if seg.pause_before_ms > 0:
                events.append(KeystrokeEvent(
                    key="", delay_before_ms=seg.pause_before_ms,
                    key_hold_ms=0, event_type=EventType.PAUSE,
                ))

            text = seg.text
            if not text:
                continue

            # Build HMM state sequence for this segment
            hmm_states = self.hmm.generate_state_sequence(len(text), initial_state=0)

            # GAN context
            ctx_vec = self.gan.build_context_vector(
                complexity=seg.complexity.value,
                fatigue=self.fatigue.get_speed_multiplier(),
                hmm_state=hmm_states[0] if hmm_states else 0,
            )
            # Sample GAN timings (we cycle through them)
            gan_timings = self.gan.sample_timings(ctx_vec, n_samples=1)[0]  # (seq_len, 3)

            prev_char = " "
            pending_error: Optional[tuple] = None  # (error_char, intended_char, extra_typed)
            extra_after_error = 0

            for i, ch in enumerate(text):
                state = hmm_states[i] if i < len(hmm_states) else 0

                # Get base timing
                if self.gan.trained:
                    gan_idx = i % len(gan_timings)
                    key_down_ms = float(gan_timings[gan_idx, 0])
                    key_hold_ms = float(gan_timings[gan_idx, 1])
                    gap_ms = float(gan_timings[gan_idx, 2])
                    delay_ms = key_down_ms + gap_ms
                else:
                    delay_ms = self.hmm.sample_delay_ms(state)
                    key_hold_ms = self.hmm.sample_hold_ms(state)

                # Apply Fitts' law correction
                fitts_ms = fitts_delay_ms(prev_char, ch)
                delay_ms = max(delay_ms, fitts_ms * 0.5) + fitts_ms * 0.3

                # Bigram speedup
                bigram_mult = self.bigram.get_speedup(prev_char, ch)
                delay_ms *= bigram_mult

                # Fatigue
                delay_ms *= self.fatigue.get_delay_multiplier()
                self.fatigue.update()
                char_count += 1

                # Special char micro-pause
                if ch in SPECIAL_CHARS:
                    delay_ms += float(np.random.uniform(30, 120))

                # Error generation
                if self.error_gen.should_make_error(state):
                    error_char, error_type = self.error_gen.generate_error(ch)
                    if error_char and error_char != ch:
                        # Type the wrong character(s)
                        for ec in error_char:
                            events.append(KeystrokeEvent(
                                key=ec,
                                delay_before_ms=max(20.0, delay_ms),
                                key_hold_ms=max(20.0, key_hold_ms),
                                event_type=EventType.KEYDOWN,
                                hmm_state=state,
                                is_error=True,
                            ))
                        # Generate correction sequence
                        correction_events = self.correction.generate_correction_events(
                            error_char, ch, base_delay_ms=delay_ms,
                        )
                        events.extend(correction_events)
                        prev_char = ch
                        continue

                # Normal keystroke
                events.append(KeystrokeEvent(
                    key=ch,
                    delay_before_ms=max(20.0, delay_ms),
                    key_hold_ms=max(20.0, key_hold_ms),
                    event_type=EventType.KEYDOWN,
                    hmm_state=state,
                ))
                prev_char = ch

        logger.info(f"Synthesized {len(events)} keystroke events")
        return events
