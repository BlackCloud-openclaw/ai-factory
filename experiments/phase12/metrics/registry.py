"""
MetricRegistry：从配置加载 Metric，支持动态导入
"""

import importlib
import inspect
import logging
from typing import Dict, List, Optional, Type

from .protocol import Metric
from ..config.benchmark import DEFAULT_METRIC_CLASSES
from ..judge.client import LLMJudgeClient

logger = logging.getLogger(__name__)


def _import_metric_class(class_path: str) -> Type[Metric]:
    """从完整导入路径导入 Metric 类"""
    if "." not in class_path:
        raise ValueError(f"Invalid metric class path: {class_path}")
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class MetricRegistry:
    def __init__(self, metrics: Optional[List[Metric]] = None):
        self._metrics: Dict[str, Metric] = {}
        if metrics:
            for m in metrics:
                self.register(m)

    def register(self, metric: Metric) -> None:
        if metric.name in self._metrics:
            raise ValueError(f"Metric '{metric.name}' already registered")
        self._metrics[metric.name] = metric

    def get(self, name: str) -> Optional[Metric]:
        return self._metrics.get(name)

    def list(self) -> List[str]:
        return list(self._metrics.keys())

    def all(self) -> List[Metric]:
        return list(self._metrics.values())

    @classmethod
    def with_defaults(cls) -> "MetricRegistry":
        """从配置加载默认 Metric，共享 LLMJudgeClient。"""
        shared_client = LLMJudgeClient()
        registry = cls()

        for class_path in DEFAULT_METRIC_CLASSES:
            try:
                metric_cls = _import_metric_class(class_path)
                # 检查是否需要注入 client
                sig = inspect.signature(metric_cls.__init__)
                if "client" in sig.parameters:
                    metric = metric_cls(client=shared_client)
                else:
                    metric = metric_cls()
                registry.register(metric)
            except Exception as e:
                logger.warning("Failed to load metric '%s': %s, skipping", class_path, e)

        return registry