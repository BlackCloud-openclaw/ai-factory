"""
Phase 15.3 — Shadow Recorder

职责：
1. 接收 ShadowRewriteResult
2. 持久化到数据库（或内存，用于测试）
3. 提供查询接口（用于实验分析）

设计原则：
- Recorder 不参与 Rewrite 或 Validator 逻辑
- 只做存储和检索
- 数据库实现可替换

A-4 修改：
- INSERT 语句增加 contract_data 字段
- 采用项目既有 JSONB 编码方式：json.dumps(..., ensure_ascii=False)

A-4 补充修复：
- 增加 _clean_json_data() 防御性清洗，处理 datetime 等不可序列化类型

Phase 15.5 修改：
- INSERT 语句增加 writer_events 字段
- writer_events 按三态语义存储：NULL/[]/[...]
"""

from typing import Protocol, Optional, runtime_checkable
from datetime import datetime, date
import json
import logging

from .result import ShadowRewriteResult, ShadowRewriteStatus

logger = logging.getLogger(__name__)


def _clean_json_data(obj):
    """
    递归清洗数据，将所有 datetime/date 转换为 ISO 字符串，
    确保 json.dumps() 不会失败。

    同时处理 Pydantic 对象和普通 dict/list。
    """
    if obj is None:
        return None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _clean_json_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json_data(v) for v in obj]
    # 如果是 Pydantic 对象，尝试转为 dict
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return _clean_json_data(obj.model_dump(mode="json"))
        except Exception:
            # 如果失败，尝试 model_dump()
            return _clean_json_data(obj.model_dump())
    # 如果是普通对象，尝试用 str() 兜底
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


@runtime_checkable
class ShadowRecorder(Protocol):
    """Shadow Recorder 协议。"""

    async def record(self, result: ShadowRewriteResult) -> None:
        """记录一个 Shadow Rewrite 结果。"""
        ...

    async def get_by_scene(self, scene_id: str, limit: int = 10):
        """按场景查询记录。"""
        ...

    async def get_by_experiment(self, experiment_id: str):
        """按实验 ID 查询记录。"""
        ...


class MemoryShadowRecorder:
    """内存版 Shadow Recorder（用于测试）。"""

    def __init__(self):
        self._records: list[ShadowRewriteResult] = []

    async def record(self, result: ShadowRewriteResult) -> None:
        self._records.append(result)
        logger.debug(f"[Shadow] Recorded result for {result.scene_id}")

    async def get_by_scene(self, scene_id: str, limit: int = 10):
        filtered = [r for r in self._records if r.scene_id == scene_id]
        return filtered[-limit:] if limit > 0 else filtered

    async def get_by_experiment(self, experiment_id: str):
        return [r for r in self._records if r.experiment_id == experiment_id]

    async def count_by_status(self, status: str) -> int:
        return sum(1 for r in self._records if r.status == status)

    async def all(self):
        return self._records


class DatabaseShadowRecorder:
    """数据库版 Shadow Recorder。"""

    def __init__(self, pool):
        self._pool = pool

    async def record(self, result: ShadowRewriteResult) -> None:
        async with self._pool.acquire() as conn:
            # 防御性清洗：确保所有 JSONB 数据可序列化
            cleaned_contract_data = _clean_json_data(result.contract_data)
            cleaned_original_violations = _clean_json_data(result.original_violations)
            cleaned_rewritten_violations = _clean_json_data(result.rewritten_violations)

            # ========== Phase 15.5: writer_events 清洗 ==========
            # 三态语义：
            # - None → 保持 None（数据不可用）
            # - [] → 保持空数组（Writer 返回空 events）
            # - [...] → 保持正常数组
            cleaned_writer_events = _clean_json_data(result.writer_events)
            # =====================================================

            await conn.execute(
                """
                INSERT INTO shadow_rewrite_log (
                    scene_id,
                    original_text,
                    rewritten_text,
                    original_length,
                    rewritten_length,
                    original_passed,
                    rewritten_passed,
                    original_violations,
                    rewritten_violations,
                    status,
                    error_message,
                    experiment_id,
                    prompt_version,
                    contract_id,
                    model,
                    executed_at,
                    contract_data,
                    writer_events                -- ========== Phase 15.5 新增 ==========
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9,
                    $10, $11, $12, $13, $14, $15, $16, $17, $18
                )
                """,
                result.scene_id,
                result.original_text,
                result.rewritten_text,
                result.original_length,
                result.rewritten_length,
                result.original_validation_passed,
                result.rewritten_validation_passed,
                (
                    json.dumps(cleaned_original_violations, ensure_ascii=False)
                    if cleaned_original_violations
                    else None
                ),
                (
                    json.dumps(cleaned_rewritten_violations, ensure_ascii=False)
                    if cleaned_rewritten_violations
                    else None
                ),
                result.status.value,
                result.error_message,
                result.experiment_id,
                result.prompt_version or "phase15.3.v1",
                result.contract_id,
                result.model,
                result.executed_at,
                (
                    json.dumps(cleaned_contract_data, ensure_ascii=False)
                    if cleaned_contract_data
                    else None
                ),
                (
                    json.dumps(cleaned_writer_events, ensure_ascii=False)
                    if cleaned_writer_events is not None  # 保持 NULL 为 NULL，不序列化
                    else None
                ),
            )

    async def get_by_scene(self, scene_id: str, limit: int = 10):
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM shadow_rewrite_log
                WHERE scene_id = $1
                ORDER BY executed_at DESC
                LIMIT $2
                """,
                scene_id, limit
            )
            return [self._row_to_result(r) for r in rows]

    async def get_by_experiment(self, experiment_id: str):
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM shadow_rewrite_log
                WHERE experiment_id = $1
                ORDER BY executed_at DESC
                """,
                experiment_id
            )
            return [self._row_to_result(r) for r in rows]

    def _row_to_result(self, row) -> ShadowRewriteResult:
        return ShadowRewriteResult(
            scene_id=row["scene_id"],
            original_text=row["original_text"],
            original_validation_passed=row["original_passed"],
            original_violations=row["original_violations"],
            rewritten_text=row["rewritten_text"],
            rewritten_validation_passed=row["rewritten_passed"],
            rewritten_violations=row["rewritten_violations"],
            status=ShadowRewriteStatus(row["status"]),
            error_message=row["error_message"],
            experiment_id=row["experiment_id"],
            prompt_version=row["prompt_version"] or "phase15.3.v1",
            contract_id=row["contract_id"],
            model=row["model"],
            executed_at=row["executed_at"],
            contract_data=row["contract_data"],
            writer_events=row["writer_events"],  # ========== Phase 15.5 新增 ==========
        )