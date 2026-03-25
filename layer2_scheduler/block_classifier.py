from typing import List
from layer2_scheduler.ast_parser import ASTNode
from layer2_scheduler.language_registry import LanguageRegistry
from layer2_scheduler.typing_plan import ComplexityLevel
from loguru import logger

LEVEL_MAP = {
    "BOILERPLATE": ComplexityLevel.BOILERPLATE,
    "SIMPLE": ComplexityLevel.SIMPLE,
    "MODERATE": ComplexityLevel.MODERATE,
    "COMPLEX": ComplexityLevel.COMPLEX,
    "VERY_COMPLEX": ComplexityLevel.VERY_COMPLEX,
}


class BlockClassifier:
    def __init__(self):
        self.registry = LanguageRegistry()

    def classify(self, nodes: List[ASTNode], language: str) -> List[tuple]:
        """Returns list of (ASTNode, ComplexityLevel)"""
        complexity_map = self.registry.get_complexity_map(language)
        results = []
        for node in nodes:
            level_str = complexity_map.get(node.node_type, "SIMPLE")
            level = LEVEL_MAP.get(level_str, ComplexityLevel.SIMPLE)
            results.append((node, level))
        return results
