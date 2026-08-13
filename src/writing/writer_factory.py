"""
Phase 12.2B: Writer Factory

提供统一的 ControlledWriter 创建入口。
"""

import logging
from typing import Optional
from src.writing.controlled_writer import ControlledWriter
from src.writing.snapshot_adapter import SnapshotWriterAdapter

logger = logging.getLogger(__name__)


def create_writer(
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    max_retries_per_segment: int = 2,
    enable_fallback: bool = True,
) -> SnapshotWriterAdapter:
    """
    创建 SnapshotWriterAdapter。

    强制使用本地 LLM 服务 (http://localhost:8082, Qwen3-32B-Q5_K_M.gguf)
    确保与 docker-compose 配置一致。
    """
    # 强制使用正确的地址和模型
    forced_api_base = "http://localhost:8082"
    forced_model = "Qwen3-32B-Q5_K_M.gguf"

    # 记录实际使用的配置
    logger.info(f"Creating Writer with api_base={forced_api_base}, model={forced_model}")

    writer = ControlledWriter(
        api_base=forced_api_base,
        model=forced_model,
        max_retries_per_segment=max_retries_per_segment,
        enable_fallback=enable_fallback,
    )
    return SnapshotWriterAdapter(writer)