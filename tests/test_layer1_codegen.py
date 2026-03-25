import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from layer1_codegen.code_buffer import CodeBuffer, DependencyInfo
from layer1_codegen.dependency_extractor import DependencyExtractor


def test_code_buffer_creation():
    buf = CodeBuffer(raw_code="print('hello')", language="python")
    assert buf.raw_code == "print('hello')"
    assert buf.language == "python"
    assert len(buf) == 14


def test_dependency_extractor_python():
    code = """
import os
from pathlib import Path

def my_function():
    pass

class MyClass:
    pass
"""
    buf = CodeBuffer(raw_code=code, language="python")
    extractor = DependencyExtractor()
    info = extractor.extract(buf)
    assert "os" in info.imports
    assert "pathlib" in info.imports
    assert "my_function" in info.functions


def test_dependency_extractor_javascript():
    code = """
import React from 'react';
const axios = require('axios');

function App() {
    return null;
}
"""
    buf = CodeBuffer(raw_code=code, language="javascript")
    extractor = DependencyExtractor()
    info = extractor.extract(buf)
    assert "react" in info.imports


def test_code_buffer_lines():
    code = "line1\nline2\nline3"
    buf = CodeBuffer(raw_code=code, language="python")
    assert buf.lines() == ["line1", "line2", "line3"]
