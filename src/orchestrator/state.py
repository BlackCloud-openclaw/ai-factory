from typing import Annotated, Any, Sequence, List, Dict, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """State definition for the AI Factory LangGraph workflow."""

    # ===== 用户输入/项目标识 =====
    user_input: str = ""
    original_request: str = ""          # 与 user_input 等效，便于 planner 使用
    project_id: str = ""                # 记忆隔离
    metadata: Dict[str, Any] = {}

    # ===== 消息历史 =====
    messages: Annotated[list, add_messages] = []

    # ===== 分析结果 =====
    intent: str = ""
    subtasks: List[str] = []            # 语意分析出的子任务描述
    is_complex: bool = False

    # ===== 知识检索 =====
    research_results: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []  # 知识来源（可选）

    # ===== 代码生成与执行 =====
    code_generated: str = ""
    code_file_path: str = ""
    execution_result: Optional[Dict[str, Any]] = None

    # ===== 验证 =====
    validation_result: Optional[Dict[str, Any]] = None

    # ===== 计划与调度 =====
    task_plan: Optional[Dict[str, Any]] = None   # 任务计划（TaskPlan 序列化）
    plan_id: str = ""                            # 计划ID
    task_id: str = ""                            # 调度器任务ID
    subtask_results: Dict[str, Any] = {}         # 子任务执行结果
    plan_status: str = ""                        # 计划执行状态

    # ===== 最终输出 =====
    final_answer: str = ""

    # ===== 重试控制 =====
    retry_count: int = 0
    max_retries: int = 3
    max_retries_per_subtask: int = 2
    step_count: int = 0
    remaining_subtasks: List = []                # 用于 advance_subtask
    current_subtask_index: int = 0
    current_subtask_id: str = ""
    needs_retry: bool = False

    # ===== 节点跟踪与错误 =====
    current_node: str = ""
    error: Optional[str] = None

    # ===== 记忆上下文 =====
    memory_context: Dict[str, Any] = {}

    # ===== 兼容旧字段（保留，逐步废弃） =====
    skip_remaining: bool = False
    plan: List[Dict[str, Any]] = []              # 旧的计划格式

    def should_retry(self) -> bool:
        return self.retry_count < self.max_retries and self.error is not None
