import asyncio
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a function with exponential backoff and jitter."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(
                            base_delay * (2**attempt) + random.uniform(0, 0.1 * base_delay),
                            max_delay,
                        )
                        time.sleep(delay)
            raise last_exception

        return wrapper

    return decorator


def smart_retry(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    fallback_model: Optional[str] = None,
    retryable_exceptions: tuple = (Exception,),
) -> Callable:
    """智能重试装饰器，支持 LLM 模型降级切换。

    重试策略：
        attempt 1: 使用原 model
        attempt 2: 使用原 model（再次尝试）
        attempt 3: 切换到 fallback_model（如果提供）

    Args:
        max_retries: 最大重试次数（不包括首次调用），默认 3
        backoff_factor: 退避因子，等待时间 = backoff_factor * (2 ** (attempt-1))
        fallback_model: 重试失败后切换的备用模型名
        retryable_exceptions: 需要重试的异常类型

    Returns:
        装饰后的函数
    """
    from src.common.logging import setup_logging

    logger = setup_logging("common.retry")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_model = kwargs.get("model", fallback_model)

            for attempt in range(max_retries + 1):
                try:
                    if current_model:
                        kwargs["model"] = current_model
                    logger.debug(
                        f"Calling {func.__name__} attempt {attempt + 1}/{max_retries + 1}"
                    )
                    result = await func(*args, **kwargs)
                    return result
                except retryable_exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}"
                    )

                    if attempt < max_retries:
                        if attempt == 2 and fallback_model:
                            current_model = fallback_model
                            logger.info(
                                f"Switching to fallback model: {fallback_model}"
                            )
                        delay = backoff_factor * (2**attempt)
                        logger.info(
                            f"Retrying in {delay:.1f}s... (attempt {attempt + 2}/{max_retries + 1})"
                        )
                        await asyncio.sleep(delay)

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_model = kwargs.get("model", fallback_model)

            for attempt in range(max_retries + 1):
                try:
                    if current_model:
                        kwargs["model"] = current_model
                    logger.debug(
                        f"Calling {func.__name__} attempt {attempt + 1}/{max_retries + 1}"
                    )
                    result = func(*args, **kwargs)
                    return result
                except retryable_exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}"
                    )

                    if attempt < max_retries:
                        if attempt == 2 and fallback_model:
                            current_model = fallback_model
                            logger.info(
                                f"Switching to fallback model: {fallback_model}"
                            )
                        delay = backoff_factor * (2**attempt)
                        logger.info(
                            f"Retrying in {delay:.1f}s... (attempt {attempt + 2}/{max_retries + 1})"
                        )
                        time.sleep(delay)

            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
