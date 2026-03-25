import pytest
from layer2_scheduler.ast_parser import ASTParser
from layer2_scheduler.block_classifier import BlockClassifier
from layer2_scheduler.klm_scheduler import KLMScheduler
from layer2_scheduler.pause_injector import PauseInjector
from layer2_scheduler.nonlinear_router import NonlinearRouter
from layer2_scheduler.typing_plan import ComplexityLevel, KLMOperator, TypingSegment
from layer2_scheduler.language_registry import LanguageRegistry
from layer1_codegen.code_buffer import CodeBuffer

KLM_CONFIG = {"K": 0.28, "M": 1.35, "H": 0.40, "P": 1.10}
PAUSE_CONFIG = {"micro_min": 2, "micro_max": 15, "short_min": 15, "short_max": 180,
                "mid_min": 180, "mid_max": 600, "long_min": 600}


def test_ast_parser_fallback_python():
    code = "import os\ndef hello():\n    pass"
    parser = ASTParser()
    nodes = parser.parse(code, "python")
    assert len(nodes) > 0
    node_types = [n.node_type for n in nodes]
    # At least one import node
    assert any("import" in t for t in node_types)


def test_block_classifier_boilerplate():
    from layer2_scheduler.ast_parser import ASTNode
    classifier = BlockClassifier()
    node = ASTNode("import_statement", "import os", 0, 9)
    result = classifier.classify([node], "python")
    assert result[0][1] == ComplexityLevel.BOILERPLATE


def test_block_classifier_complex():
    from layer2_scheduler.ast_parser import ASTNode
    classifier = BlockClassifier()
    node = ASTNode("function_definition", "def foo():\n    pass", 0, 19)
    result = classifier.classify([node], "python")
    assert result[0][1] == ComplexityLevel.COMPLEX


def test_klm_scheduler_adds_mental_op_for_complex():
    scheduler = KLMScheduler(KLM_CONFIG)
    seg = TypingSegment(
        text="def complex_function():\n    pass",
        complexity=ComplexityLevel.COMPLEX,
        pause_before_ms=0.0,
    )
    result = scheduler.schedule([seg])
    assert KLMOperator.M in result[0].klm_ops


def test_klm_scheduler_no_mental_for_boilerplate():
    scheduler = KLMScheduler(KLM_CONFIG)
    seg = TypingSegment(
        text="import os",
        complexity=ComplexityLevel.BOILERPLATE,
        pause_before_ms=0.0,
    )
    result = scheduler.schedule([seg])
    assert KLMOperator.M not in result[0].klm_ops


def test_klm_scheduler_special_char_pause():
    scheduler = KLMScheduler(KLM_CONFIG)
    seg = TypingSegment(
        text="(argument)",
        complexity=ComplexityLevel.SIMPLE,
        pause_before_ms=0.0,
    )
    result = scheduler.schedule([seg])
    assert result[0].pause_before_ms > 0  # special char ( adds pause


def test_pause_injector_micro_range():
    injector = PauseInjector(PAUSE_CONFIG)
    for _ in range(20):
        pause = injector.get_pause_ms(ComplexityLevel.BOILERPLATE)
        assert 2000 <= pause <= 15000  # 2-15 seconds in ms


def test_pause_injector_lognormal_distribution():
    import numpy as np
    injector = PauseInjector(PAUSE_CONFIG)
    pauses = [injector.get_pause_ms(ComplexityLevel.MODERATE) for _ in range(100)]
    assert np.std(pauses) > 100  # should have some variance


def test_nonlinear_router_adds_backvisits():
    import numpy as np
    np.random.seed(42)
    router = NonlinearRouter({"backvisit_probability": 1.0})  # force all backvisits
    segments = [
        TypingSegment(text=f"code_{i}", complexity=ComplexityLevel.SIMPLE)
        for i in range(5)
    ]
    result = router.route(segments)
    backvisits = [s for s in result if s.is_backvisit]
    assert len(backvisits) > 0


def test_language_registry_complexity_map():
    reg = LanguageRegistry()
    cm = reg.get_complexity_map("python")
    assert cm["function_definition"] == "COMPLEX"
    assert cm["import_statement"] == "BOILERPLATE"


def test_full_scheduler():
    from layer2_scheduler.scheduler import Layer2Scheduler
    config = {
        "klm": KLM_CONFIG,
        "pause": PAUSE_CONFIG,
        "nonlinear": {"backvisit_probability": 0.0},
    }
    scheduler = Layer2Scheduler(config)
    buf = CodeBuffer(raw_code="import os\ndef hello():\n    return 42", language="python")
    plan = scheduler.build_plan(buf)
    assert len(plan.segments) > 0
    assert plan.total_text == buf.raw_code
    assert plan.estimated_duration_ms > 0
