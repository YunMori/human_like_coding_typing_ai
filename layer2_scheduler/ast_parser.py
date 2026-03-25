from typing import List, Tuple, Optional
from layer2_scheduler.language_registry import LanguageRegistry
from loguru import logger


class ASTNode:
    def __init__(self, node_type: str, text: str, start_byte: int, end_byte: int):
        self.node_type = node_type
        self.text = text
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.children: List["ASTNode"] = []


class ASTParser:
    def __init__(self):
        self.registry = LanguageRegistry()

    def parse(self, code: str, language: str) -> List[ASTNode]:
        parser = self.registry.get_parser(language)
        if parser is None:
            return self._fallback_parse(code, language)
        try:
            tree = parser.parse(bytes(code, "utf8"))
            return self._extract_nodes(tree.root_node, code)
        except Exception as e:
            logger.warning(f"AST parse failed for {language}: {e}, using fallback")
            return self._fallback_parse(code, language)

    def _extract_nodes(self, node, code: str) -> List[ASTNode]:
        nodes = []
        if not node.is_named:
            return nodes
        text = code[node.start_byte:node.end_byte]
        ast_node = ASTNode(
            node_type=node.type,
            text=text,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
        )
        for child in node.children:
            ast_node.children.extend(self._extract_nodes(child, code))
        nodes.append(ast_node)
        return nodes

    def _fallback_parse(self, code: str, language: str) -> List[ASTNode]:
        """Line-by-line fallback when tree-sitter is unavailable."""
        nodes = []
        offset = 0
        for line in code.splitlines(keepends=True):
            stripped = line.strip()
            node_type = self._infer_node_type(stripped, language)
            nodes.append(ASTNode(
                node_type=node_type,
                text=line,
                start_byte=offset,
                end_byte=offset + len(line.encode()),
            ))
            offset += len(line.encode())
        return nodes

    def _infer_node_type(self, line: str, language: str) -> str:
        if language == "python":
            if line.startswith(("import ", "from ")):
                return "import_statement"
            elif line.startswith("def "):
                return "function_definition"
            elif line.startswith("class "):
                return "class_definition"
            elif line.startswith(("if ", "elif ")):
                return "if_statement"
            elif line.startswith("for "):
                return "for_statement"
            elif line.startswith("while "):
                return "while_statement"
            elif line.startswith("try:"):
                return "try_statement"
        elif language in ("javascript", "typescript"):
            if line.startswith("import "):
                return "import_declaration"
            elif "function " in line:
                return "function_declaration"
            elif line.startswith("class "):
                return "class_declaration"
        return "expression_statement"
