# src/capabilities/spec.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from packaging.version import Version


@dataclass(frozen=True)
class CapabilityMetadata:
    """Capability 元数据（纯数据）"""
    display_name: str
    description: str
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilitySpec:
    """
    Capability 规格说明（纯数据）

    - id: 全局唯一标识
    - version: 语义版本（packaging.version.Version）
    - metadata: 元数据
    - config_schema: 配置的 JSON Schema（可选）
    """
    id: str
    version: Version
    metadata: CapabilityMetadata
    config_schema: Optional[Dict[str, Any]] = None