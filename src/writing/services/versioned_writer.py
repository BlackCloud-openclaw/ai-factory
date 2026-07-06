"""
版本化 Writer - 并行生成 A/B/C 三个版本
Phase 6 新增，不修改现有 Writer 逻辑
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.writing.services.writing import WritingService
from src.writing.services.models import WritingCommand
from src.orchestrator.state_patch import StatePatch

logger = logging.getLogger(__name__)


@dataclass
class VersionedWritingResult:
    """三个版本 + 对比摘要"""
    version_a: Dict[str, Any]   # 无 Director
    version_b: Dict[str, Any]   # 有 Director
    version_c: Dict[str, Any]   # B + Rewrite
    comparison: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["📊 三版本对比:"]
        for ver, data in [("A", self.version_a), ("B", self.version_b), ("C", self.version_c)]:
            kpi = data.get("kpi", {})
            lines.append(f"  版本{ver}: NV={kpi.get('narrative_value', 0):.2f} | "
                         f"对话={kpi.get('dialogue', 0):.1f} | "
                         f"冲突={kpi.get('conflict', 0):.1f} | "
                         f"角色={kpi.get('character', 0):.1f}")
        if self.comparison:
            lines.append(f"  Director 增量 (B-A): +{self.comparison.get('director_gain', 0):.2f}")
            lines.append(f"  Rewrite 增量 (C-B): +{self.comparison.get('rewrite_gain', 0):.2f}")
        return "\n".join(lines)


class VersionedWriter:
    """
    并行生成三个版本的 Writer
    对外接口与 WritingAgent 兼容，但内部串联三次生成
    """

    @staticmethod
    def _safe_json_dumps(data: Dict[str, Any]) -> str:
        """
        生成 JSON 字符串并验证其有效性。
        如果验证失败，使用 ensure_ascii=True 重试。
        """
        # 第一次尝试：使用 ensure_ascii=False（保留 Unicode，更可读）
        try:
            json_str = json.dumps(data, ensure_ascii=False)
            json.loads(json_str)  # 验证
            return json_str
        except Exception as e:
            logger.warning(f"JSON dumps with ensure_ascii=False failed: {e}, retrying with ensure_ascii=True")
            # 第二次尝试：使用 ensure_ascii=True（转义所有非 ASCII 字符）
            try:
                json_str = json.dumps(data, ensure_ascii=True)
                json.loads(json_str)  # 验证
                logger.info("JSON regeneration with ensure_ascii=True succeeded")
                return json_str
            except Exception as e2:
                logger.error(f"JSON dumps completely failed: {e2}")
                # 最终 fallback：极简 JSON
                fallback = {
                    "scene_text": str(data.get("scene_text", "生成失败"))[:500],
                    "events": [],
                    "foreshadowing": [],
                }
                return json.dumps(fallback, ensure_ascii=True)

    @staticmethod
    async def generate_versions(
        novel_id: str,
        volume: int,
        chapter: int,
        scene_idx: int,
        scene_plan: Dict[str, Any],
        current_state: Dict[str, Any],
        writing_feedback: str = "",
        voiceprint_config_path: Optional[str] = None,
        save_to_db: bool = True,
        # 新增参数
        narrative_blueprint: Optional[Dict[str, Any]] = None,
        knowledge_deltas: Optional[List[Dict[str, Any]]] = None,
        character_intent: Optional[Dict[str, Any]] = None,
        # ====== 新增：戏剧结构（来自 Drama Planner） ======
        drama_structure: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,  # 新增
    ) -> VersionedWritingResult:
        """
        生成三个版本并保存
        """
        logger.info(f"[VersionedWriter] generate_versions called for scene {scene_idx}")
        
        versions = {}

        # ----- 版本 A：无 Director（基线） -----
        logger.info(f"[VersionedWriter] 生成版本 A (无Director) - 场景 {scene_idx}")
        cmd_a = WritingCommand(
            novel_id=novel_id,
            volume=volume,
            chapter=chapter,
            scene_idx=scene_idx,
            scene_plan=scene_plan,
            current_state=current_state,
            writing_feedback=writing_feedback,
            voiceprint_config_path=voiceprint_config_path,
            narrative_blueprint=None,
            knowledge_deltas=None,
            character_intent=None,
            drama_structure=drama_structure,
            metadata=metadata,  # 传递
        )
        result_a = await WritingService.execute(cmd_a)

        a_json = VersionedWriter._safe_json_dumps({
            "scene_text": result_a.scene_text or "",
            "events": result_a.events or [],
            "foreshadowing": [],
        })

        versions["A"] = {
            "scene_text": a_json,
            "events": result_a.events,
            "kpi": _compute_kpi(result_a.scene_text, {}, {}),
        }

        # ----- 版本 B：有 Director -----
        logger.info(f"[VersionedWriter] 生成版本 B (有Director) - 场景 {scene_idx}")
        #director_output = await _call_director(scene_plan, current_state, novel_id, volume, chapter)

        cmd_b = WritingCommand(
            novel_id=novel_id,
            volume=volume,
            chapter=chapter,
            scene_idx=scene_idx,
            scene_plan=scene_plan,
            current_state=current_state,
            writing_feedback=writing_feedback,
            voiceprint_config_path=voiceprint_config_path,
            # 不再传递 Director 输出
            narrative_blueprint=None,
            knowledge_deltas=None,
            character_intent=None,
            # ====== 传递戏剧结构 ======
            drama_structure=drama_structure,
            metadata=metadata,  # 添加
        )
        result_b = await WritingService.execute(cmd_b)

        b_json = VersionedWriter._safe_json_dumps({
            "scene_text": result_b.scene_text or "",
            "events": result_b.events or [],
            "foreshadowing": [],
        })

        versions["B"] = {
            "scene_text": b_json,
            "events": result_b.events,
            "kpi": _compute_kpi(result_b.scene_text, {}, {}),
        }

        # ----- 版本 C：B + Rewrite -----
        logger.info(f"[VersionedWriter] 生成版本 C (B + Rewrite) - 场景 {scene_idx}")

        # 从版本 B 的 JSON 中提取纯文本 scene_text
        try:
            b_data = json.loads(versions["B"]["scene_text"])
            b_plain_text = b_data.get("scene_text", "")
        except Exception:
            b_plain_text = versions["B"]["scene_text"]
            logger.warning(f"[VersionedWriter] Failed to parse B scene_text as JSON, using raw string")

        rewritten_text = await _call_rewrite(b_plain_text)

        c_json = VersionedWriter._safe_json_dumps({
            "scene_text": rewritten_text or "",
            "events": versions["B"].get("events", []),
            "foreshadowing": [],
        })

        versions["C"] = {
            "scene_text": c_json,
            "events": versions["B"].get("events", []),
            "kpi": _compute_kpi(rewritten_text, {}, {}),
        }

        # ----- 计算对比摘要 -----
        comparison = {
            "director_gain": versions["B"]["kpi"].get("narrative_value", 0) -
                             versions["A"]["kpi"].get("narrative_value", 0),
            "rewrite_gain": versions["C"]["kpi"].get("narrative_value", 0) -
                            versions["B"]["kpi"].get("narrative_value", 0),
        }

        # ----- 保存到数据库 -----
        if save_to_db:
            await _save_versions_to_db(novel_id, volume, chapter, scene_idx, versions)

        return VersionedWritingResult(
            version_a=versions["A"],
            version_b=versions["B"],
            version_c=versions["C"],
            comparison=comparison,
        )


# ---------- 辅助函数 ----------
def _compute_kpi(scene_text: str, state_before: Dict, state_after: Dict) -> Dict:
    """计算单场景 KPI（延迟导入避免循环）"""
    from src.writing.narrative_kpi import NarrativeKPIEngine
    engine = NarrativeKPIEngine()
    result = engine.compute(scene_text, state_before, state_after)
    return result.to_dict()


async def _call_rewrite(scene_text: str) -> str:
    """调用 Rewrite Agent（延迟导入）"""
    from src.agents.rewrite import RewriteAgent
    from src.orchestrator.state import AgentState

    temp_state = AgentState(scene_text=scene_text)
    agent = RewriteAgent()
    result = await agent.run(temp_state)
    return result.get("polished_text", scene_text)


async def _save_versions_to_db(
    novel_id: str,
    volume: int,
    chapter: int,
    scene_idx: int,
    versions: Dict[str, Dict],
):
    """保存三个版本到 narrative_versions 表"""
    import json
    from src.db import get_db_pool

    pool = get_db_pool()
    if not pool:
        logger.warning("Database pool unavailable, skipping version save")
        return

    async with pool.acquire() as conn:
        for version_type, data in versions.items():
            await conn.execute("""
                INSERT INTO narrative_versions
                (novel_id, volume_num, chapter_num, scene_idx, version_type,
                 scene_text, world_state, kpi_scores)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (novel_id, volume_num, chapter_num, scene_idx, version_type)
                DO UPDATE SET
                    scene_text = EXCLUDED.scene_text,
                    world_state = EXCLUDED.world_state,
                    kpi_scores = EXCLUDED.kpi_scores,
                    generated_at = NOW()
            """, novel_id, volume, chapter, scene_idx, version_type,
                data["scene_text"],
                json.dumps({}),
                json.dumps(data.get("kpi", {})),
            )
    logger.info(f"[VersionedWriter] Saved 3 versions for {novel_id} v{volume}c{chapter}s{scene_idx}")