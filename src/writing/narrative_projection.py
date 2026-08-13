"""
Phase 13.2: NarrativeProjection Continuity Anchor

NarrativeProjection 是跨章节叙事控制状态，由 Runtime 维护，
不由 LLM 直接生成或改写。
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from src.writing.narrative_intent import SceneRole
import hashlib


class NarrativeProjection(BaseModel):
    """
    当前故事阶段的叙事投影。

    不描述世界事实，描述读者应该持续关注的叙事压力。
    这是结构化 IR，不是自然语言摘要。

    注意：active_conflict 和 next_pressure 应始终有值，
    但在初始状态或尚未建立时允许为 None。
    """

    projection_id: str = Field(..., description="投影唯一标识")
    chapter_id: str = Field(..., description="关联章节 ID")

    # 当前主线压力
    active_conflict: Optional[str] = Field(
        None, description="当前主要叙事压力，如 '师门隐藏真实目的'。应始终有值。"
    )

    # 未关闭的问题
    unresolved_threads: List[str] = Field(
        default_factory=list, description="未解决的叙事线索列表"
    )

    # 当前阶段目标
    active_objectives: List[str] = Field(
        default_factory=list, description="当前阶段需要完成的目标"
    )

    # 当前情绪方向
    emotional_state: Optional[str] = Field(
        None, description="当前情绪状态，如 '怀疑增加但未决裂'"
    )

    # 下一章节压力方向
    next_pressure: Optional[str] = Field(
        None, description="下一章必须面对的叙事压力。应始终有值。"
    )

    # 最近执行意图
    last_intent_id: str = Field(..., description="最近一次 NarrativeIntent 的 ID")

    # 最近场景角色
    last_scene_role: SceneRole = Field(
        ..., description="最近完成场景的 SceneRole"
    )

    # 单调递增版本，每次更新 +1
    version: int = Field(1, ge=1, description="投影版本号，单调递增")

    # 元数据
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: int) -> int:
        if v < 1:
            raise ValueError("version must be >= 1")
        return v

    def to_dict(self) -> dict:
        """序列化为字典"""
        data = self.model_dump(mode="json")
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "NarrativeProjection":
        """从字典反序列化"""
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls.model_validate(data)

    @classmethod
    def generate_projection_id(cls, chapter_id: str, last_intent_id: str) -> str:
        """确定性生成 projection_id"""
        raw = f"{chapter_id}|{last_intent_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def increment_version(self) -> "NarrativeProjection":
        """返回一个版本号 +1 的副本（使用 Pydantic model_copy）"""
        return self.model_copy(
            update={
                "version": self.version + 1,
                "updated_at": datetime.now()
            }
        )
# ========== 临时存根：保留旧 NarrativeProjector 功能 ==========
class NarrativeProjector:
    """旧叙事投影器（Phase 6）存根"""
    @staticmethod
    async def get_latest(novel_id: str):
        """获取最新投影（旧功能）"""
        return None

    @staticmethod
    async def project(novel_id, event, chapter, event_id, last_projection=None):
        """执行投影（旧功能）"""
        return None
