from typing import Annotated, Any, List, Dict, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from src.orchestrator.state_patch import WorkflowPhase
from src.common.canonical import canonical_hash  

class AgentState(BaseModel):
    """State definition for the AI Factory LangGraph workflow."""

    # ===== 用户输入/项目标识 =====
    user_input: str = ""
    original_request: str = ""          # 与 user_input 等效，便于 planner 使用
    project_id: str = ""                # 记忆隔离
    novel_id: Optional[str] = None      # 当前小说ID（用于写作场景）
    metadata: Dict[str, Any] = {}       # 仅用于临时、非结构化数据
    state_hash: Optional[str] = None   # 新增：状态哈希，用于审计

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
    max_retries_per_subtask: int = 3
    step_count: int = 0
    remaining_subtasks: List = []                # 用于 advance_subtask
    current_subtask_index: int = 0
    current_subtask_id: str = ""
    needs_retry: bool = False

    # ===== 节点跟踪与错误 =====
    current_node: str = ""
    error: Optional[str] = None

    # ===== 验证模式 =====
    validation_mode: str = "code"   # 代码验证模式（默认）或 novel

    # ===== 记忆上下文 =====
    memory_context: Dict[str, Any] = {}

    # ===== 事件溯源与状态缓存（用于小说写作）=====
    pending_tool_calls: List[Dict[str, Any]] = []   # 待处理的工具调用
    applied_events: List[Any] = []                  # 已应用的事件（可选）
    current_state: Dict[str, Any] = {}              # 当前世界状态缓存
    last_sequence_id: int = 0                       # 最新事件 sequence_id
    compressed_state: Optional[Dict[str, Any]] = None   # 卷级别压缩状态，包含角色意图等
    voice_memory: Optional[Dict[str, Any]] = None   # 存储 VoiceFingerprint 的字典

    # ===== 小说写作专用字段（结构化）=====
    chapter_id: Optional[str] = None
    resume: bool = False                     # 是否为断点续写
    task_type: str = "code"                  # code, novel_outline, scene_plan
    outline: Optional[Dict[str, Any]] = None # 小说大纲
    current_volume: int = 1
    current_chapter: int = 1
    scene_plan: Optional[Dict[str, Any]] = None   # 当前场景计划
    scene_plan_list: List[Dict[str, Any]] = Field(default_factory=list)
    total_chapters_in_volume: int = 0
    total_scenes_in_chapter: int = 0
    current_scene_index: Optional[int] = 0   # 0‑based 已完成场景数（下一个要生成的场景索引）
    writing_constraints: Optional[Dict[str, Any]] = None
    scene_text: str = ""

    # Director 输出字段（阶段2新增）
    narrative_blueprint: Optional[Dict[str, Any]] = None
    knowledge_deltas: Optional[List[Dict[str, Any]]] = None
    character_intent: Optional[Dict[str, Any]] = None
    
    # ====== 新增：Drama Planner 输出（取代 Director） ======
    drama_structure: Optional[Dict[str, Any]] = None

    # ===== 工作流阶段（显式状态机）=====
    phase: Optional[WorkflowPhase] = None

    # ===== 写作反馈（用于重试）=====
    writing_feedback: str = ""

    # ===== 临时诊断字段（逐步废弃，保留兼容）=====
    deviation_detected: bool = False
    missing_goal_keywords: List[str] = Field(default_factory=list)
    missing_conflict_keywords: List[str] = Field(default_factory=list)

    # ===== 已废弃字段（保留以免反序列化失败，但不再使用）=====
    skip_remaining: bool = False         # 不再使用
    plan: List[Dict[str, Any]] = []      # 不再使用

    # ====== 新增：Planning Contract ======
    planning_contract: Optional[Dict[str, Any]] = None  # 当前场景的 Planning Contract
    planning_contracts: List[Dict[str, Any]] = Field(default_factory=list)  # 所有场景的 Contracts

    def should_retry(self) -> bool:
        return self.retry_count < self.max_retries and self.error is not None
    
    def compute_state_hash(self) -> str:
        """计算 AgentState 的确定性哈希（忽略非持久化字段）"""
        # 选择影响叙事状态的字段，忽略 step_count, retry_count 等运行时计数
        hash_fields = {
            "novel_id": self.novel_id,
            "current_volume": self.current_volume,
            "current_chapter": self.current_chapter,
            "current_scene_index": self.current_scene_index,
            "current_state": self.current_state,
            "outline": self.outline,
            "scene_plan_list": self.scene_plan_list,
            "scene_plan": self.scene_plan,
            "compressed_state": self.compressed_state,
            "phase": self.phase.value if self.phase else None,
            "last_sequence_id": self.last_sequence_id,
            "total_scenes_in_chapter": self.total_scenes_in_chapter,
            "total_chapters_in_volume": self.total_chapters_in_volume,
            # 忽略容易变化的字段：messages, research_results, code_generated, execution_result, validation_result 等
        }
        return canonical_hash(hash_fields)    