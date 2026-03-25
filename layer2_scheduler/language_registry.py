from typing import Optional, Dict
from loguru import logger

try:
    from tree_sitter_languages import get_parser as ts_get_parser
    TS_LANGUAGES_AVAILABLE = True
except ImportError:
    TS_LANGUAGES_AVAILABLE = False
    logger.warning("tree-sitter-languages not available, using fallback parser")

SUPPORTED_LANGUAGES = {
    "python", "javascript", "typescript", "java", "c", "cpp",
    "go", "rust", "ruby", "php", "swift", "kotlin", "bash", "sql"
}

COMPLEXITY_NODES: Dict[str, Dict[str, str]] = {
    "python": {
        "import_statement": "BOILERPLATE",
        "import_from_statement": "BOILERPLATE",
        "pass_statement": "BOILERPLATE",
        "function_definition": "COMPLEX",
        "class_definition": "COMPLEX",
        "if_statement": "MODERATE",
        "for_statement": "MODERATE",
        "while_statement": "MODERATE",
        "try_statement": "COMPLEX",
        "with_statement": "MODERATE",
        "lambda": "MODERATE",
        "decorator": "MODERATE",
        "async_function_definition": "VERY_COMPLEX",
        "comprehension": "COMPLEX",
    },
    "javascript": {
        "import_statement": "BOILERPLATE",
        "import_declaration": "BOILERPLATE",
        "function_declaration": "COMPLEX",
        "arrow_function": "COMPLEX",
        "class_declaration": "COMPLEX",
        "if_statement": "MODERATE",
        "for_statement": "MODERATE",
        "while_statement": "MODERATE",
        "try_statement": "COMPLEX",
        "async_function": "VERY_COMPLEX",
        "await_expression": "MODERATE",
        "template_literal": "SIMPLE",
    },
    "typescript": {
        "import_statement": "BOILERPLATE",
        "import_declaration": "BOILERPLATE",
        "function_declaration": "COMPLEX",
        "arrow_function": "COMPLEX",
        "class_declaration": "COMPLEX",
        "interface_declaration": "COMPLEX",
        "type_alias_declaration": "MODERATE",
        "if_statement": "MODERATE",
        "for_statement": "MODERATE",
        "try_statement": "COMPLEX",
        "async_function": "VERY_COMPLEX",
        "generic_type": "COMPLEX",
    },
    "java": {
        "import_declaration": "BOILERPLATE",
        "package_declaration": "BOILERPLATE",
        "method_declaration": "COMPLEX",
        "class_declaration": "COMPLEX",
        "interface_declaration": "COMPLEX",
        "if_statement": "MODERATE",
        "for_statement": "MODERATE",
        "while_statement": "MODERATE",
        "try_statement": "COMPLEX",
        "lambda_expression": "VERY_COMPLEX",
        "annotation": "SIMPLE",
    },
    "go": {
        "import_declaration": "BOILERPLATE",
        "package_clause": "BOILERPLATE",
        "function_declaration": "COMPLEX",
        "method_declaration": "COMPLEX",
        "type_declaration": "MODERATE",
        "if_statement": "MODERATE",
        "for_statement": "MODERATE",
        "select_statement": "COMPLEX",
        "go_statement": "VERY_COMPLEX",
        "defer_statement": "MODERATE",
        "interface_type": "COMPLEX",
    },
    "rust": {
        "use_declaration": "BOILERPLATE",
        "function_item": "COMPLEX",
        "impl_item": "COMPLEX",
        "trait_item": "COMPLEX",
        "if_expression": "MODERATE",
        "match_expression": "COMPLEX",
        "while_expression": "MODERATE",
        "for_expression": "MODERATE",
        "closure_expression": "COMPLEX",
        "async_block": "VERY_COMPLEX",
        "macro_invocation": "MODERATE",
    },
}

# Default for unlisted languages
DEFAULT_COMPLEXITY_NODES = {
    "function": "COMPLEX",
    "class": "COMPLEX",
    "if": "MODERATE",
    "for": "MODERATE",
    "while": "MODERATE",
    "import": "BOILERPLATE",
}


class LanguageRegistry:
    _parsers: Dict[str, any] = {}

    def get_parser(self, language: str):
        if not TS_LANGUAGES_AVAILABLE:
            return None
        lang = language.lower()
        if lang == "cpp":
            lang = "cpp"
        elif lang == "c":
            lang = "c"
        if lang not in self._parsers:
            try:
                self._parsers[lang] = ts_get_parser(lang)
                logger.debug(f"Loaded tree-sitter parser for {lang}")
            except Exception as e:
                logger.warning(f"Could not load parser for {lang}: {e}")
                self._parsers[lang] = None
        return self._parsers[lang]

    def get_complexity_map(self, language: str) -> dict:
        return COMPLEXITY_NODES.get(language.lower(), DEFAULT_COMPLEXITY_NODES)

    def is_supported(self, language: str) -> bool:
        return language.lower() in SUPPORTED_LANGUAGES
