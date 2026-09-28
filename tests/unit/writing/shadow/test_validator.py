# src/writing/bootstrap/composition_root.py

import logging
import os
from dataclasses import dataclass
from typing import Optional, Any

from src.capabilities.runtime import FrozenRuntimeCapabilityRegistry
from src.writing.runtime.services import RuntimeServices
from src.writing.runtime import ValidationPolicy
from src.config import settings
from src.db import get_db_pool

# ========== 原有导入 ==========
from src.writing.bootstrap.runtime_capabilities import build_runtime_capabilities
# 如果有 build_runtime_services 则导入，否则直接构建
# 根据你之前的代码，应该有 build_runtime_services 或类似函数
# 如果没有，我们可以直接构建 RuntimeServices
from src.writing.runtime.services import RuntimeServices

# ========== Phase 15.3 Shadow 导入 ==========
from src.writing.shadow import (
    ShadowRewriteRunner,
    ShadowRewriter,
    ShadowValidator,
    LLMClient,
    DatabaseShadowRecorder,
    MemoryShadowRecorder,
    ShadowPromptBuilder,
)
from src.writing.validation.semantic_validator import SemanticValidator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WriterRuntime:
    """Writer Runtime 环境。"""

    runtime_capabilities: FrozenRuntimeCapabilityRegistry
    runtime_services: RuntimeServices
    validation_policy: ValidationPolicy

    # Phase 15.3 — Shadow
    shadow_enabled: bool = False
    shadow_sample_ratio: float = 0.0
    shadow_experiment_id: str = "phase15.3.v1"
    shadow_runner: Optional[Any] = None
    shadow_recorder: Optional[Any] = None
    shadow_tasks: Optional[set] = None


def build_writer_runtime() -> WriterRuntime:
    """
    构建 Writer Runtime 环境。

    包含：
    - Runtime Capabilities
    - Runtime Services
    - Validation Policy
    - Phase 15.3 Shadow 组件（默认关闭）
    """

    # ========== 1. 构建 Runtime Capabilities ==========
    capabilities = build_runtime_capabilities()

    # ========== 2. 构建 Runtime Services ==========
    # 根据你现有的代码，RuntimeServices 接受 capabilities
    # 也可能有其他参数，但最基本的是 capabilities
    services = RuntimeServices(capabilities)

    # ========== 3. 构建 Validation Policy ==========
    # 根据环境变量选择策略
    env = os.getenv("ENVIRONMENT", "development")
    if env == "production":
        validation_policy = ValidationPolicy.production()
    else:
        validation_policy = ValidationPolicy.development()

    # ========== 4. Phase 15.3 — Shadow ==========
    shadow_enabled = getattr(settings, 'shadow_rewrite_enabled', False)
    shadow_sample_ratio = getattr(settings, 'shadow_rewrite_sample_ratio', 0.0)
    shadow_experiment_id = getattr(settings, 'shadow_rewrite_experiment_id', 'phase15.3.v1')

    shadow_runner = None
    shadow_recorder = None
    shadow_tasks = None

    if shadow_enabled:
        # 4.1 LLM Client
        api_base = getattr(settings, 'shadow_llm_api_base', None) or settings.llm_api_url
        model = getattr(settings, 'shadow_llm_model', None) or settings.llm_model_name
        temperature = getattr(settings, 'shadow_llm_temperature', 0.3)
        max_tokens = getattr(settings, 'shadow_llm_max_tokens', 4096)
        timeout = getattr(settings, 'shadow_llm_timeout', 120.0)

        llm_client = LLMClient(
            api_base=api_base,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        # 4.2 Prompt Builder
        prompt_builder = ShadowPromptBuilder()

        # 4.3 Rewriter
        rewriter = ShadowRewriter(llm_client, prompt_builder)

        # 4.4 Validator（生产 SemanticValidator）
        semantic_validator = SemanticValidator()
        validator = ShadowValidator(semantic_validator)

        # 4.5 Runner
        shadow_runner = ShadowRewriteRunner(
            rewriter=rewriter,
            validator=validator,
        )

        # 4.6 Recorder
        pool = get_db_pool()
        if pool is not None:
            shadow_recorder = DatabaseShadowRecorder(pool)
        else:
            shadow_recorder = MemoryShadowRecorder()
            logger.warning("[Shadow] No DB pool available, using MemoryShadowRecorder")

        # 4.7 Task tracking (保留为 None，暂不实现)
        shadow_tasks = None

        logger.info(
            "[Shadow] Runtime initialized: sample_ratio=%.2f, experiment=%s",
            shadow_sample_ratio,
            shadow_experiment_id,
        )

    # ========== 5. 组装并返回 ==========
    return WriterRuntime(
        runtime_capabilities=capabilities,
        runtime_services=services,
        validation_policy=validation_policy,
        shadow_enabled=shadow_enabled,
        shadow_sample_ratio=shadow_sample_ratio,
        shadow_experiment_id=shadow_experiment_id,
        shadow_runner=shadow_runner,
        shadow_recorder=shadow_recorder,
        shadow_tasks=shadow_tasks,
    )