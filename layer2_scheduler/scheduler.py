from typing import List
from layer1_codegen.code_buffer import CodeBuffer
from layer2_scheduler.ast_parser import ASTParser
from layer2_scheduler.block_classifier import BlockClassifier
from layer2_scheduler.klm_scheduler import KLMScheduler
from layer2_scheduler.nonlinear_router import NonlinearRouter
from layer2_scheduler.pause_injector import PauseInjector
from layer2_scheduler.typing_plan import TypingPlan, TypingSegment, ComplexityLevel
from loguru import logger


class Layer2Scheduler:
    def __init__(self, config: dict):
        self.ast_parser = ASTParser()
        self.classifier = BlockClassifier()
        self.klm = KLMScheduler(config.get("klm", {}))
        self.router = NonlinearRouter(config.get("nonlinear", {}))
        self.pauser = PauseInjector(config.get("pause", {}))

    def build_plan(self, buffer: CodeBuffer) -> TypingPlan:
        code = buffer.raw_code
        language = buffer.language

        # Parse AST
        nodes = self.ast_parser.parse(code, language)

        if nodes:
            # Classify by complexity
            classified = self.classifier.classify(nodes, language)
            segments = []
            for node, complexity in classified:
                pause = self.pauser.get_pause_ms(complexity)
                seg = TypingSegment(
                    text=node.text,
                    complexity=complexity,
                    pause_before_ms=pause,
                    node_type=node.node_type,
                )
                segments.append(seg)
        else:
            # Fallback: treat whole code as one segment
            segments = [TypingSegment(
                text=code,
                complexity=ComplexityLevel.MODERATE,
                pause_before_ms=0.0,
            )]

        # Apply KLM scheduling
        segments = self.klm.schedule(segments)

        # Apply nonlinear routing
        segments = self.router.route(segments)

        # Estimate duration
        total_chars = sum(len(s.text) for s in segments)
        base_wpm = 65
        estimated_ms = (total_chars / 5) / base_wpm * 60 * 1000
        estimated_ms += sum(s.pause_before_ms for s in segments)

        logger.info(f"Built plan: {len(segments)} segments, ~{estimated_ms/1000:.1f}s estimated")

        return TypingPlan(
            segments=segments,
            total_text=code,
            estimated_duration_ms=estimated_ms,
            language=language,
        )
