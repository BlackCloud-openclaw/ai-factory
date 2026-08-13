"""
Architecture Test: 确保 ControlledWriter 只能通过 RuntimeServices 注入实例化。
使用 AST 解析，支持白名单配置。
"""

import ast
import pytest
from pathlib import Path

# 扫描目录
SCAN_DIRS = ["src"]

# 排除的目录/文件
EXCLUDE_DIRS = ["tests", "__pycache__", ".pytest_cache"]

# 白名单：允许直接实例化 ControlledWriter 的文件（相对路径）
# 在这些文件中，实例化是合理的（如定义工厂、组合根等）
ALLOWED_INSTANTIATION_FILES = {
    Path("src/orchestrator/nodes.py"),  # writer_node 内部通过 runtime 参数注入
    # 未来如果新增 composition_root 或 factory，可加入白名单
}

CLASS_DEF_NAME = "ControlledWriter"


def _is_controlled_writer_call(node: ast.Call) -> bool:
    """检查 Call 节点是否调用了 ControlledWriter 构造函数"""
    func = node.func
    if isinstance(func, ast.Name) and func.id == CLASS_DEF_NAME:
        return True
    if isinstance(func, ast.Attribute) and func.attr == CLASS_DEF_NAME:
        return True
    return False


def _has_runtime_services_kwarg(node: ast.Call) -> bool:
    """检查调用是否包含 runtime_services 关键字参数"""
    return any(kw.arg == "runtime_services" for kw in node.keywords)


def _node_in_class(node: ast.AST, class_node: ast.ClassDef) -> bool:
    """检查 node 是否位于 class_node 的 body 内"""
    for child in ast.walk(class_node):
        if child == node:
            return True
    return False


def test_controlled_writer_instantiation_uses_runtime_services():
    """
    扫描所有业务代码，检查 ControlledWriter() 调用是否包含了 runtime_services 参数。
    若发现裸调用（无 runtime_services 参数），则测试失败。
    """
    violations = []

    for dir_name in SCAN_DIRS:
        base_dir = Path(dir_name)
        if not base_dir.exists():
            continue

        for py_file in base_dir.rglob("*.py"):
            # 排除目录
            if any(excl in str(py_file) for excl in EXCLUDE_DIRS):
                continue

            # 检查白名单
            is_allowed = False
            for allowed_path in ALLOWED_INSTANTIATION_FILES:
                if str(py_file).endswith(str(allowed_path)):
                    is_allowed = True
                    break
            if is_allowed:
                continue

            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except SyntaxError:
                continue  # 跳过语法不完整的文件

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue

                # 如果调用发生在 ControlledWriter 类定义内部，跳过
                parent_class = None
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef) and parent.name == CLASS_DEF_NAME:
                        if _node_in_class(node, parent):
                            parent_class = parent.name
                            break
                if parent_class == CLASS_DEF_NAME:
                    continue

                if _is_controlled_writer_call(node):
                    if not _has_runtime_services_kwarg(node):
                        violations.append(
                            f"{py_file}:{node.lineno} - 裸调用 ControlledWriter()，缺少 runtime_services 参数"
                        )

    assert not violations, (
        "违反 Runtime 注入约束：\n" + "\n".join(violations) +
        "\n请使用 `ControlledWriter(runtime_services=...)` 替代裸实例化。"
    )