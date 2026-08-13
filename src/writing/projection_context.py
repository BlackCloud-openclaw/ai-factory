"""
Phase 13.2: ProjectionContext

Planner 消费的 Projection 视图，避免 Prompt 膨胀。
"""

from typing import Optional, List
from pydantic import BaseModel
from src.writing.narrative_projection import NarrativeProjection


class ProjectionContext(BaseModel):
    """
    Planner 实际消费的叙事状态视图。
    从 NarrativeProjection 提取关键信息，控制 Prompt 长度。
    """

    active_conflict: Optional[str]
    unresolved_threads: List[str]
    next_pressure: Optional[str]

    @classmethod
    def from_projection(cls, projection: NarrativeProjection) -> "ProjectionContext":
        """从 NarrativeProjection 构建视图"""
        # 限制线程数量，避免 Prompt 膨胀
        threads = projection.unresolved_threads[:5] if projection.unresolved_threads else []

        return cls(
            active_conflict=projection.active_conflict,
            unresolved_threads=threads,
            next_pressure=projection.next_pressure,
        )

    def to_prompt_text(self) -> str:
        """生成 Prompt 文本"""
        lines = ["## 当前叙事状态 (Narrative Projection)"]
        lines.append(f"Active Conflict: {self.active_conflict if self.active_conflict else '无'}")
        lines.append("Unresolved Threads:")
        if self.unresolved_threads:
            for t in self.unresolved_threads:
                lines.append(f"  - {t}")
        else:
            lines.append("  无")
        lines.append(f"Next Pressure: {self.next_pressure if self.next_pressure else '无'}")
        lines.append("")
        lines.append("你的 NarrativeIntent 必须回应这些未完成的叙事线。")
        return "\n".join(lines)