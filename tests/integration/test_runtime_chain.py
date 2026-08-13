"""
Runtime 注入链路测试（AST 静态检查）
验证 writer_node 内部实例化 ControlledWriter 时传入了 runtime_services 参数。
不导入任何模块。
"""

import ast
import pytest
from pathlib import Path


def test_runtime_services_passed_to_controlled_writer_in_ast():
    """检查 writer_node 中是否有 `ControlledWriter(runtime_services=...)` 调用"""
    nodes_file = Path("src/orchestrator/nodes.py")
    assert nodes_file.exists(), f"{nodes_file} 不存在"

    tree = ast.parse(nodes_file.read_text(encoding="utf-8"))

    # 查找 writer_node 函数体
    writer_node_body = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "writer_node":
            writer_node_body = node
            break

    assert writer_node_body is not None, "未找到 writer_node 函数定义"

    # 在函数体中查找 `ControlledWriter(` 调用
    found = False
    for node in ast.walk(writer_node_body):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "ControlledWriter":
            # 检查关键字参数中是否有 runtime_services
            has_runtime_services = any(kw.arg == "runtime_services" for kw in node.keywords)
            if has_runtime_services:
                found = True
                break

    assert found, "writer_node 中未找到 ControlledWriter(runtime_services=...) 调用"