import asyncio
import json
import hashlib
import time
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass

from openai import AsyncOpenAI
import httpx

from .models import JudgeResult, JudgeDimension, JudgeCacheKey, JudgeCacheEntry
from .prompts import (
    CONTINUITY_JUDGE_PROMPT,
    CHARACTER_JUDGE_PROMPT,
    DIALOGUE_JUDGE_PROMPT,
    FLOW_JUDGE_PROMPT,
    PROMPT_VERSIONS,
)
from ..config.benchmark import (
    JUDGE_MODEL,
    JUDGE_API_BASE,
    JUDGE_MAX_CONCURRENCY,
    JUDGE_CACHE_TTL,
)


@dataclass(frozen=True)
class JudgeConfig:
    model: str = JUDGE_MODEL
    api_base: str = JUDGE_API_BASE
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    cache_enabled: bool = True
    cache_ttl: int = JUDGE_CACHE_TTL
    max_concurrency: int = JUDGE_MAX_CONCURRENCY


class LLMJudgeClient:
    """LLM Judge 客户端（带缓存、限流、共享连接）"""

    PROMPT_TEMPLATES = {
        JudgeDimension.CONTINUITY: CONTINUITY_JUDGE_PROMPT,
        JudgeDimension.CHARACTER: CHARACTER_JUDGE_PROMPT,
        JudgeDimension.DIALOGUE: DIALOGUE_JUDGE_PROMPT,
        JudgeDimension.FLOW: FLOW_JUDGE_PROMPT,
    }

    def __init__(
        self,
        config: Optional[JudgeConfig] = None,
        cache: Optional[Dict[JudgeCacheKey, JudgeCacheEntry]] = None,
    ):
        self._config = config or JudgeConfig()
        self._cache = cache or {}
        self._cache_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrency)
        self._client = AsyncOpenAI(
            api_key="not-needed",
            base_url=self._config.api_base,
            timeout=httpx.Timeout(self._config.timeout_seconds, connect=10.0),
        )

    async def evaluate(
        self,
        dimension: JudgeDimension,
        text: str,
        context: Dict[str, Any],
        use_cache: bool = True,
    ) -> JudgeResult:
        async with self._semaphore:
            return await self._evaluate_internal(dimension, text, context, use_cache)

    async def _evaluate_internal(
        self,
        dimension: JudgeDimension,
        text: str,
        context: Dict[str, Any],
        use_cache: bool,
    ) -> JudgeResult:
        prompt_version = PROMPT_VERSIONS.get(dimension, "1.0")
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        cache_key = JudgeCacheKey(
            dimension=dimension.value,
            text_hash=text_hash,
            prompt_version=prompt_version,
            model=self._config.model,
        )

        # 缓存读取（带锁）
        if use_cache and self._config.cache_enabled:
            async with self._cache_lock:
                entry = self._cache.get(cache_key)
                if entry and time.time() - entry.timestamp < entry.ttl:
                    return entry.result

        prompt = self._build_prompt(dimension, text, context)

        start_time = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            raw_response = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return JudgeResult(
                dimension=dimension,
                score=0.5,
                confidence=0.0,
                reasoning=f"LLM call failed: {e}",
                tokens_used=0,
                elapsed_ms=elapsed_ms,
                raw_response={"error": str(e)},
            )

        try:
            data = self._parse_response(raw_response)
            result = self._to_judge_result(dimension, data, tokens_used, elapsed_ms, raw_response)
        except Exception as e:
            result = JudgeResult(
                dimension=dimension,
                score=0.5,
                confidence=0.0,
                reasoning=f"Parse error: {e}",
                tokens_used=tokens_used,
                elapsed_ms=elapsed_ms,
                raw_response={"raw": raw_response[:500], "error": str(e)},
            )

        # 缓存写入（带锁）
        if use_cache and self._config.cache_enabled:
            async with self._cache_lock:
                self._cache[cache_key] = JudgeCacheEntry(
                    result=result,
                    timestamp=time.time(),
                    ttl=self._config.cache_ttl,
                )

        return result