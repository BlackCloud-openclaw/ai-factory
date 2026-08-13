# tests/architecture/test_runtime_boundaries.py

import ast
import pytest
from pathlib import Path

RUNTIME_DIR = Path("src/runtime")

# Loader 是唯一允许使用兼容层和 matcher 的模块
LOADER_MODULE = "loader.py"


def _iter_py_files(directory: Path, exclude_loader: bool = False):
    if not directory.exists():
        return
    for py_file in directory.rglob("*.py"):
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        if exclude_loader and py_file.name == LOADER_MODULE:
            continue
        yield py_file


def test_runtime_does_not_import_builtin_capability():
    """Runtime 模块不得 import src.capabilities.builtin.*"""
    for py_file in _iter_py_files(RUNTIME_DIR):
        with open(py_file, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("src.capabilities.builtin"):
                        raise AssertionError(f"{py_file}: imports {alias.name}")
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("src.capabilities.builtin"):
                    raise AssertionError(f"{py_file}: imports from {node.module}")


def test_runtime_does_not_import_capability_registry():
    """Runtime 不应直接导入 CapabilityRegistry（应使用 CapabilityLookup）"""
    for py_file in _iter_py_files(RUNTIME_DIR):
        if py_file.name in ["loader.py", "__init__.py"]:
            continue

        with open(py_file, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "src.capabilities":
                    for alias in node.names:
                        if alias.name == "CapabilityRegistry":
                            raise AssertionError(f"{py_file}: imports CapabilityRegistry")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "src.capabilities.CapabilityRegistry":
                        raise AssertionError(f"{py_file}: imports CapabilityRegistry")


def test_runtime_does_not_import_surface_compatibility():
    """Runtime 不得导入 surfaces.compatibility，Loader 除外"""
    for py_file in _iter_py_files(RUNTIME_DIR, exclude_loader=True):
        with open(py_file, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "surfaces.compatibility" in node.module:
                    raise AssertionError(f"{py_file}: imports {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "surfaces.compatibility" in alias.name:
                        raise AssertionError(f"{py_file}: imports {alias.name}")


def test_runtime_does_not_access_matcher_attribute():
    """Runtime 不应访问 .matcher，Loader 除外"""
    for py_file in _iter_py_files(RUNTIME_DIR, exclude_loader=True):
        with open(py_file, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr == "matcher":
                    raise AssertionError(f"{py_file}:{node.lineno}: accesses '.matcher'")