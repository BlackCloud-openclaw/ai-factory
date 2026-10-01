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
    import sys
    print("[DIAG] build_writer_runtime ENTERED", file=sys.stderr)
    sys.stderr.flush()
    """
    构建 Writer Runtime 环境。

    包含：
    - Runtime Capabilities
    - Runtime Services (含 Rewriter)
    - Validation Policy
    - Phase 15.3 Shadow 组件（默认关闭）
    """
    print("[Shadow] === build_writer_runtime called (print) ===")
    logger.info("[Shadow] === build_writer_runtime called (logger) ===")

    # ========== 1. 构建 Runtime Capabilities ==========
    capabilities = build_runtime_capabilities()

    # ========== Phase 15.7-A: 构建生产 Rewriter ==========
    print(f"[DIAG] literary_rewrite_enabled = {getattr(settings, 'literary_rewrite_enabled', False)}", file=sys.stderr)
    sys.stderr.flush()
    logger.critical("[15.7-A] Checking literary_rewrite_enabled: %s", getattr(settings, 'literary_rewrite_enabled', False))
    production_rewriter = None
    if getattr(settings, 'literary_rewrite_enabled', False):
        print("[DIAG] Entering Rewriter build block", file=sys.stderr)
        sys.stderr.flush()
        try:
            logger.critical("[15.7-A] Building production Rewriter...")
            api_base = getattr(settings, 'shadow_llm_api_base', None) or "http://localhost:8082"
            model = getattr(settings, 'shadow_llm_model', None) or "Qwen3-32B-Q5_K_M-writer"
            temperature = getattr(settings, 'shadow_llm_temperature', 0.3)
            max_tokens = getattr(settings, 'shadow_llm_max_tokens', 4096)
            timeout = getattr(settings, 'shadow_llm_timeout', 120.0)

            logger.critical("[15.7-A] LLMClient params: api_base=%s, model=%s", api_base, model)
            llm_client = LLMClient(
                api_base=api_base,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            prompt_builder = ShadowPromptBuilder(prompt_version="phase15.7.v1")
            production_rewriter = ShadowRewriter(llm_client, prompt_builder)
            logger.critical("[15.7-A] production_rewriter built successfully: type=%s", type(production_rewriter).__name__)
        except Exception as e:
            logger.critical("[15.7-A] Failed to build production Rewriter: %s", e, exc_info=True)
            production_rewriter = None
    else:
        logger.critical("[15.7-A] literary_rewrite_enabled is False, skipping Rewriter build")

    # ========== 2. 构建 Runtime Services（注入 Rewriter） ==========
    logger.critical("[15.7-A] Passing production_rewriter to RuntimeServices: type=%s", type(production_rewriter).__name__ if production_rewriter else "None")
    print("[DIAG] Creating RuntimeServices...", file=sys.stderr)
    sys.stderr.flush()
    services = RuntimeServices(capabilities, rewriter=production_rewriter)
    logger.critical("[15.7-A] RuntimeServices created, rewriter attribute: type=%s", type(services.rewriter).__name__ if services.rewriter else "None")

    # ========== Phase 15.8-fix P0-12: 构建共享 SemanticValidator ==========
    # 与 ValidatorAgent._get_validator() 采用相同的构造参数
    # 目标：让 ControlledWriter 的 _validate_segment 走 embedding 匹配，
    #       避免 QualityGate.score 恒为 0.00 → 3 次重试 → force_pass
    semantic_validator = None
    try:
        _provider = LocalHttpEmbeddingProvider(
            endpoint=_config.embedding_endpoint,
            dim=_config.embedding_dim,
        )
        semantic_validator = SemanticValidator(
            embedding_provider=_provider,
            keyword_threshold=0.6,
            embedding_threshold=0.30,
            embedding_min_confidence=0.6,
            enable_embedding=getattr(_config, 'enable_embedding_validator', True),
        )
        logger.info(
            "[SharedValidator] built SemanticValidator (endpoint=%s, dim=%s)",
            _config.embedding_endpoint,
            _config.embedding_dim,
        )
    except Exception as _e:
        logger.error(
            "[SharedValidator] build failed, ControlledWriter will fall back to NoOp: %s",
            _e,
            exc_info=True,
        )
        semantic_validator = None
    # =====================================================================

    # ========== 3. 构建 Validation Policy ==========

    # ========== 3. 构建 Validation Policy ==========
    env = os.getenv("ENVIRONMENT", "development")
    if env == "production":
        validation_policy = ValidationPolicy.production()
    else:
        validation_policy = ValidationPolicy.development()

    logger.info(f"[Shadow] DEBUG: settings.shadow_rewrite_enabled = {settings.shadow_rewrite_enabled}")
    logger.info(f"[Shadow] DEBUG: settings.shadow_rewrite_sample_ratio = {settings.shadow_rewrite_sample_ratio}")

    # ========== 4. Phase 15.3 — Shadow ==========
    shadow_enabled = True
    shadow_sample_ratio = 1.0
    shadow_experiment_id = getattr(settings, 'shadow_rewrite_experiment_id', 'phase15.3.v1')

    shadow_runner = None
    shadow_recorder = None
    shadow_tasks = None

    if shadow_enabled:
        print("[Shadow] === SHADOW BLOCK ENTERED ===")
        try:
            # ========== Phase 15.7-A: 复用生产 Rewriter ==========
            if production_rewriter is not None:
                shadow_rewriter = production_rewriter
                logger.info("[Shadow] Reusing production Rewriter instance")
            else:
                # 创建独立的 Shadow Rewriter
                api_base = "http://localhost:8082"
                model = "Qwen3-32B-Q5_K_M-writer"
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
                prompt_builder = ShadowPromptBuilder()
                shadow_rewriter = ShadowRewriter(llm_client, prompt_builder)
            # =====================================================

            # Validator（生产 SemanticValidator）
            semantic_validator = SemanticValidator()
            validator = ShadowValidator(semantic_validator)

            # Runner
            shadow_runner = ShadowRewriteRunner(
                rewriter=shadow_rewriter,
                validator=validator,
            )

            # Recorder
            pool = get_db_pool()
            if pool is not None:
                shadow_recorder = DatabaseShadowRecorder(pool)
            else:
                shadow_recorder = MemoryShadowRecorder()
                logger.warning("[Shadow] No DB pool available, using MemoryShadowRecorder")

            shadow_tasks = None

            logger.info(
                "[Shadow] Runtime initialized: sample_ratio=%.2f, experiment=%s",
                shadow_sample_ratio,
                shadow_experiment_id,
            )
            print("[Shadow] === SHADOW INITIALIZATION COMPLETE ===")

        except Exception as e:
            print(f"[Shadow] === SHADOW INITIALIZATION FAILED: {e} ===")
            import traceback
            traceback.print_exc()
            raise
    print(f"[DIAG] Returning WriterRuntime with shadow_enabled={shadow_enabled}, rewriter is None? {production_rewriter is None}", file=sys.stderr)
    sys.stderr.flush()
    # ========== 5. 组装并返回 ==========
    return WriterRuntime(
        runtime_capabilities=capabilities,
        runtime_services=services,
        validation_policy=validation_policy,
        semantic_validator=semantic_validator,   # ← 新增
        shadow_enabled=shadow_enabled,
        shadow_sample_ratio=shadow_sample_ratio,
        shadow_experiment_id=shadow_experiment_id,
        shadow_runner=shadow_runner,
        shadow_recorder=shadow_recorder,
        shadow_tasks=shadow_tasks,
    )