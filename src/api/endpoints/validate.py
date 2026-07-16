"""
章节合规验证端点 - Phase 6 Runtime
"""

import json
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.runtime import validate_draft, execute_with_diagnosis
from src.common.logging import setup_logging

logger = setup_logging("api.validate")
router = APIRouter(prefix="/validate", tags=["validation"])


# ============================================================
# 请求/响应模型
# ============================================================

class ValidateDraftRequest(BaseModel):
    text: str = Field(..., description="待验证的文本内容")
    layer_targets: Optional[Dict[str, str]] = Field(
        default=None,
        description="层目标配置，默认使用 enhanced 级别"
    )


class LayerResultResponse(BaseModel):
    layer: str
    compliant: bool
    target_level: str
    evidence_count: int


class ValidateDraftResponse(BaseModel):
    compliance: float
    layer_results: list
    ir_hash: str
    sentence_count: int
    pattern_count: int
    pattern_types: list


class ExecuteDiagnosisRequest(BaseModel):
    text: str = Field(..., description="待诊断的文本内容")
    layer_targets: Optional[Dict[str, str]] = None


class ExecuteDiagnosisResponse(BaseModel):
    compliance: float
    layer_results: list
    ir_hash: str
    needs_revision: bool
    revision_plan: Optional[Dict[str, Any]] = None


# ============================================================
# 端点定义
# ============================================================

@router.post("/draft", response_model=ValidateDraftResponse)
async def validate_draft_endpoint(request: ValidateDraftRequest):
    """
    验证一段文本的合规性

    返回各层的合规状态和整体合规率
    """
    try:
        result = validate_draft(request.text, request.layer_targets)
        return ValidateDraftResponse(**result)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diagnose", response_model=ExecuteDiagnosisResponse)
async def execute_diagnosis_endpoint(request: ExecuteDiagnosisRequest):
    """
    执行完整诊断，包括合规验证和修订计划生成

    返回合规报告和可执行的修订计划
    """
    try:
        result = execute_with_diagnosis(request.text, request.layer_targets)
        return ExecuteDiagnosisResponse(**result)
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chapter/{novel_id}/{volume_num}/{chapter_num}")
async def validate_chapter(
    novel_id: str,
    volume_num: int,
    chapter_num: int,
    layer_targets: Optional[Dict[str, str]] = None
):
    """
    验证指定章节的合规性（从数据库读取）

    注：此端点需要读取数据库中的章节内容
    """
    from src.db.pool import get_db_pool

    try:
        pool = get_db_pool()
        if not pool:
            raise HTTPException(status_code=503, detail="Database pool not available")

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT scene_text FROM narrative_versions "
                "WHERE novel_id = $1 AND volume_num = $2 AND chapter_num = $3 "
                "ORDER BY scene_idx LIMIT 1",
                novel_id, volume_num, chapter_num
            )

        if not row:
            raise HTTPException(status_code=404, detail="Chapter not found")

        text = row["scene_text"]
        result = validate_draft(text, layer_targets)
        return ValidateDraftResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chapter validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))