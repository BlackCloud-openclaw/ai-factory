import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.db import get_db_pool
from src.orchestrator.state_patch import StatePatch, WorkflowPhase

logger = logging.getLogger(__name__)


@dataclass
class ChapterTransitionCommand:
    """章节切换事务的输入"""
    novel_id: str
    current_volume: int
    current_chapter: int
    total_chapters_in_volume: int
    # 可选：下一章的 outline 信息等


@dataclass
class ChapterTransitionResult:
    """章节切换事务的输出"""
    state_patch: StatePatch
    volume_finished: bool = False
    error: Optional[str] = None


class ChapterTransitionService:
    """
    章节切换事务：处理章节完成后的状态迁移。
    - 递增当前章
    - 重置场景索引
    - 清空场景计划列表
    - 如果卷结束，递增当前卷并重置章为1
    - 更新数据库进度
    """
    @staticmethod
    async def execute(cmd: ChapterTransitionCommand) -> ChapterTransitionResult:
        pool = get_db_pool()
        if not pool:
            return ChapterTransitionResult(
                state_patch=StatePatch(error="Database pool unavailable"),
                error="No db pool"
            )

        new_chapter = cmd.current_chapter + 1
        new_volume = cmd.current_volume
        volume_finished = False

        # 检查是否完成当前卷
        if cmd.total_chapters_in_volume > 0chapter_transition.py and new_chapter > cmd.total_chapters_in_volume:
            new_volume = cmd.current_volume + 1
            new_chapter = 1
            volume_finished = True
            logger.info(f"📚 Volume {cmd.current_volume} completed! Moving to volume {new_volume}")

        # 更新 writing_progress
        async with pool.acquire() as conn:
            async with conn.transaction():
                if volume_finished:
                    await conn.execute(
                        """
                        UPDATE writing_progress
                        SET current_volume = $1, current_chapter = $2, current_scene = 0, chapter_completed = FALSE, last_updated = NOW()
                        WHERE project_id = $3
                        """,
                        new_volume, new_chapter, cmd.novel_id
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE writing_progress
                        SET current_chapter = $1, current_scene = 0, chapter_completed = FALSE, last_updated = NOW()
                        WHERE project_id = $2
                        """,
                        new_chapter, cmd.novel_id
                    )

        # 构建 StatePatch
        patch = StatePatch(
            current_chapter=new_chapter,
            current_volume=new_volume,
            current_scene_index=0,
            scene_plan_list=[],
            total_scenes_in_chapter=0,
            phase=WorkflowPhase.PLANNING,  # 切换后进入规划阶段
        )

        logger.info(f"Chapter transition: {cmd.current_chapter} -> {new_chapter} (volume {cmd.current_volume} -> {new_volume})")
        return ChapterTransitionResult(
            state_patch=patch,
            volume_finished=volume_finished,
        )