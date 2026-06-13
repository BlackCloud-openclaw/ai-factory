import time
import functools
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

def timed(service_name: str = None):
    """装饰器：记录异步函数执行耗时"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                name = service_name or func.__name__
                logger.info(f"[TIMING] {name} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                name = service_name or func.__name__
                logger.error(f"[TIMING] {name} failed after {elapsed:.3f}s: {e}")
                raise
        return wrapper
    return decorator