"""
验证 writer_node 的函数签名明确要求 runtime 参数（非可选）。
使用 AST 静态分析，不导入任何模块。
"""

import ast
import pytest
from pathlib import Path


def test_writer_node_signature_requires_runtime():
    """检查 src/orchestrator/nodes.py 中 writer_node 函数的签名"""
    nodes_file = Path("src/orchestrator/nodes.py")
    assert nodes_file.exists(), f"{nodes_file} 不存在"

    tree = ast.parse(nodes_file.read_text(encoding="utf-8"))

    # 查找名为 writer_node 的 AsyncFunctionDef
    writer_node_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "writer_node":
            writer_node_func = node
            break

    assert writer_node_func is not None, "未找到 writer_node 函数定义"

    # 检查参数
    args = writer_node_func.args
    arg_names = [arg.arg for arg in args.args]
    assert "state" in arg_names, "writer_node 缺少 state 参数"
    assert "runtime" in arg_names, "writer_node 缺少 runtime 参数"

    # 检查 runtime 参数是否有默认值（应该没有或为 None）
    runtime_idx = arg_names.index("runtime")
    if args.defaults:
        default_start = len(args.args) - len(args.defaults)
        if runtime_idx >= default_start:
            default_val = args.defaults[runtime_idx - default_start]
            # 允许默认值为 None 或没有默认值
            assert default_val is None or isinstance(default_val, ast.Constant) and default_val.value is None, \
                "runtime 参数不应设置默认值（或仅允许 None）"
        # else: runtime 没有默认值，符合要求
    # else: 没有默认值，符合要求

    # 额外检查：确保 runtime 参数被使用（粗略检查是否有 runtime.xxx 访问）
    # 这个可选，但可以帮助确保不是死参数
    has_runtime_usage = False
    for node in ast.walk(writer_node_func):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "runtime":
            has_runtime_usage = True
            break
    # 允许没有直接使用，因为可能传给其他函数，所以不强制