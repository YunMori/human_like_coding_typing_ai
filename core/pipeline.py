import asyncio
import uuid
from datetime import datetime
from typing import Optional
from loguru import logger

from core.session import SessionConfig, SessionResult
from core.exceptions import TypingAIError
from layer1_codegen.code_buffer import CodeBuffer
from layer1_codegen.dependency_extractor import DependencyExtractor
from layer2_scheduler.scheduler import Layer2Scheduler
from layer3_dynamics.timing_synthesizer import TimingSynthesizer
from layer4_injection.injector_factory import InjectorFactory
from layer3_dynamics.keystroke_event import EventType


class TypingPipeline:
    def __init__(self, config: dict, model_dir: str = "models"):
        self.config = config
        self.dep_extractor = DependencyExtractor()
        self.scheduler = Layer2Scheduler(config)
        self.synthesizer = TimingSynthesizer(config, model_dir=model_dir)

    async def run(self, session: SessionConfig) -> SessionResult:
        session_id = str(uuid.uuid4())[:8]
        logger.info(f"[{session_id}] Starting pipeline: {session.prompt[:50]}")

        # Layer 1: Use prompt as raw code (code generation handled by Swift vvs)
        code_buffer = CodeBuffer(raw_code=session.prompt, language=session.language)
        self.dep_extractor.extract(code_buffer)
        logger.info(f"[{session_id}] Code: {len(code_buffer)} chars")

        # Layer 2: AST Scheduling
        logger.info(f"[{session_id}] Layer 2: Building typing plan...")
        typing_plan = self.scheduler.build_plan(code_buffer)
        logger.info(f"[{session_id}] Plan: {len(typing_plan.segments)} segments, ~{typing_plan.estimated_duration_ms/1000:.1f}s")

        # Layer 3: Timing Synthesis
        logger.info(f"[{session_id}] Layer 3: Synthesizing keystrokes...")
        events = self.synthesizer.synthesize(typing_plan)

        # Compute stats
        error_count = sum(1 for e in events if e.is_error)
        correction_count = sum(1 for e in events if e.is_correction)
        keystrokes = [e for e in events if e.event_type == EventType.KEYDOWN]
        total_delay = sum(e.delay_before_ms for e in events)
        total_keystrokes = len(keystrokes)
        error_rate = error_count / max(1, total_keystrokes)

        minutes = total_delay / 1000.0 / 60.0
        avg_wpm = (total_keystrokes / 5.0) / max(0.001, minutes)

        result = SessionResult(
            session_id=session_id,
            prompt=session.prompt,
            language=session.language,
            generated_code=code_buffer.raw_code,
            total_keystrokes=total_keystrokes,
            total_duration_ms=total_delay,
            error_count=error_count,
            correction_count=correction_count,
            error_rate=error_rate,
            avg_wpm=avg_wpm,
        )

        if not session.dry_run:
            # Layer 4: Injection
            logger.info(f"[{session_id}] Layer 4: Injecting keystrokes...")
            injector = InjectorFactory.create(session.target, {
                "dry_run": session.dry_run,
                **self.config,
            })
            await injector.setup(url=session.url, selector=session.selector)
            await injector.inject(events)
            await injector.teardown()
        else:
            logger.info(f"[{session_id}] Dry run: skipping injection")

        result.finalize()
        logger.info(f"[{session_id}] Done. {total_keystrokes} keystrokes, {error_rate:.1%} error rate, {avg_wpm:.0f} WPM")
        return result

    def generate_timing_plan(self, code: str, language: str, seed: Optional[int] = None) -> dict:
        """Layer 2~3만 실행. JSON 직렬화 가능한 dict 반환."""
        import numpy as np
        if seed is not None:
            np.random.seed(seed)

        buffer = CodeBuffer(raw_code=code, language=language)
        self.dep_extractor.extract(buffer)
        plan = self.scheduler.build_plan(buffer)
        events = self.synthesizer.synthesize(plan)
        return self._events_to_dict(events)

    def _events_to_dict(self, events) -> dict:
        keystrokes = [e for e in events if e.event_type == EventType.KEYDOWN]
        total_delay = sum(e.delay_before_ms for e in events)
        error_count = sum(1 for e in events if e.is_error)
        minutes = total_delay / 60000.0
        avg_wpm = (len(keystrokes) / 5.0) / max(0.001, minutes)
        return {
            "events": [
                {
                    "key": e.key,
                    "delay_before_ms": round(e.delay_before_ms, 2),
                    "key_hold_ms": round(e.key_hold_ms, 2),
                    "event_type": e.event_type.value,
                    "is_error": e.is_error,
                    "is_correction": e.is_correction,
                }
                for e in events
            ],
            "stats": {
                "total_keystrokes": len(keystrokes),
                "total_duration_ms": round(total_delay, 1),
                "error_rate": round(error_count / max(1, len(keystrokes)), 4),
                "avg_wpm": round(avg_wpm, 1),
            },
        }
