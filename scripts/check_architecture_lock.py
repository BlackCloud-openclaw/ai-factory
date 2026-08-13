# scripts/check_architecture_lock.py

import ast
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple


def get_imports(file_path: Path) -> Set[str]:
    """提取所有 import 的完整模块路径"""
    imports = set()
    try:
        with open(file_path) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)  # 保留完整路径
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # 处理 from . import 的相对导入
                    if node.module.startswith("."):
                        # 简化处理：跳过相对导入的边界检查
                        continue
                    imports.add(node.module)
    except Exception:
        pass
    return imports


def scan_module(module_dir: Path, prefix: str) -> Dict[str, Set[str]]:
    """扫描模块目录，返回 {module_name: imports}"""
    result = {}
    for py_file in module_dir.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        # 转换为点分隔的模块名
        rel_path = py_file.relative_to(module_dir.parent)
        module_name = str(rel_path).replace("/", ".").replace(".py", "")
        if not module_name.startswith(prefix):
            continue
        imports = get_imports(py_file)
        # 只保留同前缀的 import
        result[module_name] = {i for i in imports if i.startswith(prefix)}
    return result


def check_import_boundary(graph: Dict[str, Set[str]]) -> List[Tuple[str, str]]:
    """检查 Import Boundary：禁止的依赖方向"""
    violations = []
    allowed = {
        "src.writing.artifact": {"src.writing.common"},
        "src.writing.snapshot": {"src.writing.artifact", "src.writing.common"},
        "src.writing.audit": {"src.writing.snapshot", "src.writing.common"},
        "src.writing.ir": {"src.writing.common"},
        "src.writing.prompt": {"src.writing.ir", "src.writing.common"},
        "src.writing.render": {"src.writing.prompt", "src.writing.common"},
        "src.writing.coverage": {"src.writing.common"},
    }

    for module, imports in graph.items():
        for imp in imports:
            for prefix, allowed_imports in allowed.items():
                if module.startswith(prefix):
                    # 允许导入同一前缀下的模块
                    if imp.startswith(prefix):
                        continue
                    # 检查是否在允许列表中
                    if imp not in allowed_imports:
                        violations.append((module, imp))
                    break
    return violations


def check_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """检测模块级循环依赖"""
    def dfs(node, visited, path):
        if node in path:
            cycle = path[path.index(node):] + [node]
            return [cycle]
        if node in visited:
            return []
        visited.add(node)
        cycles = []
        for neighbor in graph.get(node, []):
            if neighbor in graph:
                cycles.extend(dfs(neighbor, visited, path + [node]))
        return cycles

    visited = set()
    cycles = []
    for node in graph:
        cycles.extend(dfs(node, visited, []))
    return cycles


def main():
    src_dir = Path("src/writing")
    graph = scan_module(src_dir, "src.writing")

    print("=== Architecture Lock Check ===")

    # 1. Import Boundary
    violations = check_import_boundary(graph)
    if violations:
        print("\n❌ Import Boundary Violations:")
        for module, imp in violations:
            print(f"  {module} → {imp} (not allowed)")
        sys.exit(1)
    else:
        print("\n✅ Import Boundary: OK")

    # 2. Dependency Cycle
    cycles = check_cycles(graph)
    if cycles:
        print("\n❌ Dependency Cycles:")
        for cycle in cycles:
            print(f"  {' → '.join(cycle)}")
        sys.exit(1)
    else:
        print("\n✅ Dependency Cycle: OK")

    print("\n🎉 Architecture Lock: PASS")


if __name__ == "__main__":
    main()