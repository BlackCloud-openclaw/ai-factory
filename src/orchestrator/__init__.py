# src/orchestrator/__init__.py

from src.orchestrator.state import AgentState
# 删除下面两行：
# from src.orchestrator.graph import create_workflow, compile_workflow

# 改为延迟导入
__all__ = ["AgentState", "create_workflow", "compile_workflow"]


def create_workflow(*args, **kwargs):
    from src.orchestrator.graph import create_workflow as _create_workflow
    return _create_workflow(*args, **kwargs)


def compile_workflow(*args, **kwargs):
    from src.orchestrator.graph import compile_workflow as _compile_workflow
    return _compile_workflow(*args, **kwargs)