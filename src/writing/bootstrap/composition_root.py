# src/writing/bootstrap/composition_root.py

import logging
import os
from dataclasses import dataclass
from typing import Optional, Any

from src.capabilities.runtime import FrozenRuntimeCapabilityRegistry
from src.writing.runtime.services import RuntimeServices
from src.writing.runtime import ValidationPolicy
from src.db import get_db_pool
from src.config.settings import Settings
settings = Settings()

# ========== 原有导入 ==========
from src.writing.bootstrap.runtime_capabilities import build_runtime_capabilities
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
from src.writing.validation.local_embedding_provider import LocalHttpEmbeddingProvider
from src.config import config as _config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WriterRuntime:
    """Writer Runtime 环境。"""

    runtime_capabilities: FrozenRuntimeCapabilityRegistry
    runtime_services: RuntimeServices
    validation_policy: ValidationPolicy

    # Phase 15.8-fix P0-12: 共享 SemanticValidator（带 embedding）
    # 由 build_writer_runtime 单次构建，Writer / ControlledWriter 共用，
    # 避免各自 new 一个 NoOp 版本导致 QualityGate 恒为 0.00
    semantic_validator: Optional[Any] = None

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
    - Runtime Services (含 Rewriter)
    - Validation Policy
    - Phase 15.3 Shadow 组件（可配置关闭）
    """
    logger.info("[Runtime] building WriterRuntime")

    # ========== 1. 构建 Runtime Capabilities ==========
    capabilities = build_runtime_capabilities()

    # ========== 2. 构建生产 Rewriter ==========
    production_rewriter = None
    if getattr(settings, 'literary_rewrite_enabled', False):
        try:
            api_base = getattr(settings, 'shadow_llm_api_base', None) or "http://localhost:8082"
            model = getattr(settings, 'shadow_llm_model', None) or "Qwen3-32B-Q5_K_M-writer"
            llm_client = LLMClient(
                api_base=api_base,
                model=model,
                temperature=getattr(settings, 'shadow_llm_temperature', 0.3),
                max_tokens=getattr(settings, 'shadow_llm_max_tokens', 4096),
                timeout=getattr(settings, 'shadow_llm_timeout', 120.0),
            )
            prompt_builder = ShadowPromptBuilder(prompt_version="phase15.7.v1")
            production_rewriter = ShadowRewriter(llm_client, prompt_builder)
            logger.info("[Runtime] production Rewriter built (model=%s)", model)
        except Exception as e:
            logger.error("[Runtime] failed to build production Rewriter: %s", e, exc_info=True)
            production_rewriter = None

    # ========== 3. 构建 Runtime Services ==========
    services = RuntimeServices(capabilities, rewriter=production_rewriter)

    # ========== 4. 共享 SemanticValidator（P0-12） ==========
    shared_semantic_validator = None
    try:
        _provider = LocalHttpEmbeddingProvider(
            endpoint=_config.embedding_endpoint,
            dim=_config.embedding_dim,
        )
        shared_semantic_validator = SemanticValidator(
            embedding_provider=_provider,
            keyword_threshold=0.6,
            embedding_threshold=0.30,
            embedding_min_confidence=0.6,
            enable_embedding=getattr(_config, 'enable_embedding_validator', True),
        )
        logger.info(
            "[Runtime] shared SemanticValidator built (endpoint=%s, dim=%s)",
            _config.embedding_endpoint,
            _config.embedding_dim,
        )
    except Exception as e:
        logger.error("[Runtime] shared SemanticValidator build failed: %s", e, exc_info=True)
        shared_semantic_validator = None

    # ========== 5. Validation Policy ==========
    env = os.getenv("ENVIRONMENT", "development")
    validation_policy = (
        ValidationPolicy.production() if env == "production"
        else ValidationPolicy.development()
    )

    # ========== 6. Shadow（可配置） ==========
    shadow_enabled = bool(getattr(settings, 'shadow_rewrite_enabled', False))
    shadow_sample_ratio = float(getattr(settings, 'shadow_rewrite_sample_ratio', 0.0))
    shadow_experiment_id = getattr(settings, 'shadow_rewrite_experiment_id', 'phase15.3.v1')

    shadow_runner = None
    shadow_recorder = None
    shadow_tasks = None

    if shadow_enabled:
        try:
            # 复用生产 Rewriter；没有则独立构建
            if production_rewriter is not None:
                shadow_rewriter = production_rewriter
            else:
                llm_client = LLMClient(
                    api_base="http://localhost:8082",
                    model="Qwen3-32B-Q5_K_M-writer",
                    temperature=getattr(settings, 'shadow_llm_temperature', 0.3),
                    max_tokens=getattr(settings, 'shadow_llm_max_tokens', 4096),
                    timeout=getattr(settings, 'shadow_llm_timeout', 120.0),
                )
                shadow_rewriter = ShadowRewriter(llm_client, ShadowPromptBuilder())

            # Shadow 使用独立的 validator（不污染共享实例）
            shadow_semantic_validator = shared_semantic_validator or SemanticValidator()
            shadow_validator = ShadowValidator(shadow_semantic_validator)

            shadow_runner = ShadowRewriteRunner(
                rewriter=shadow_rewriter,
                validator=shadow_validator,
            )

            pool = get_db_pool()
            if pool is not None:
                shadow_recorder = DatabaseShadowRecorder(pool)
            else:
                shadow_recorder = MemoryShadowRecorder()
                logger.warning("[Runtime] no DB pool, using MemoryShadowRecorder")

            logger.info(
                "[Runtime] Shadow enabled: ratio=%.2f, experiment=%s",
                shadow_sample_ratio,
                shadow_experiment_id,
            )
        except Exception as e:
            # 降级：Shadow 失败不阻塞生产
            logger.error("[Runtime] Shadow init failed, disabled: %s", e, exc_info=True)
            shadow_enabled = False
            shadow_runner = None
            shadow_recorder = None

    # ========== 7. 组装 ==========
    return WriterRuntime(
        runtime_capabilities=capabilities,
        runtime_services=services,
        validation_policy=validation_policy,
        semantic_validator=shared_semantic_validator,   # 共享实例，不再被 Shadow 覆盖
        shadow_enabled=shadow_enabled,
        shadow_sample_ratio=shadow_sample_ratio,
        shadow_experiment_id=shadow_experiment_id,
        shadow_runner=shadow_runner,
        shadow_recorder=shadow_recorder,
        shadow_tasks=shadow_tasks,
    )