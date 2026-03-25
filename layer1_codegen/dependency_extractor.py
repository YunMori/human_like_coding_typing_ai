import re
from layer1_codegen.code_buffer import CodeBuffer, DependencyInfo
from loguru import logger


IMPORT_PATTERNS = {
    "python": [r"^import\s+(\S+)", r"^from\s+(\S+)\s+import"],
    "javascript": [r"^import\s+.*from\s+['\"]([^'\"]+)['\"]", r"^const\s+\w+\s*=\s*require\(['\"]([^'\"]+)['\"]\)"],
    "typescript": [r"^import\s+.*from\s+['\"]([^'\"]+)['\"]"],
    "java": [r"^import\s+(\S+);"],
    "go": [r"\"([^\"]+)\""],
    "rust": [r"^use\s+(\S+);"],
}

FUNCTION_PATTERNS = {
    "python": r"^def\s+(\w+)\s*\(",
    "javascript": r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()",
    "typescript": r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()",
    "java": r"(?:public|private|protected|static|\s)+\w+\s+(\w+)\s*\(",
    "go": r"^func\s+(\w+)\s*\(",
    "rust": r"^fn\s+(\w+)\s*\(",
}


class DependencyExtractor:
    def extract(self, buffer: CodeBuffer) -> DependencyInfo:
        lang = buffer.language.lower()
        imports = self._extract_imports(buffer.raw_code, lang)
        functions = self._extract_functions(buffer.raw_code, lang)
        buffer.dependency_info = DependencyInfo(imports=imports, functions=functions)
        logger.debug(f"Extracted {len(imports)} imports, {len(functions)} functions")
        return buffer.dependency_info

    def _extract_imports(self, code: str, lang: str) -> list:
        patterns = IMPORT_PATTERNS.get(lang, [])
        imports = []
        for line in code.splitlines():
            for pat in patterns:
                m = re.search(pat, line.strip())
                if m:
                    imports.append(m.group(1))
                    break
        return list(set(imports))

    def _extract_functions(self, code: str, lang: str) -> list:
        pat = FUNCTION_PATTERNS.get(lang)
        if not pat:
            return []
        funcs = []
        for line in code.splitlines():
            m = re.search(pat, line.strip())
            if m:
                name = next((g for g in m.groups() if g), None)
                if name:
                    funcs.append(name)
        return funcs
