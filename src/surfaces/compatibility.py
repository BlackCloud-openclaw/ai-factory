# src/surfaces/compatibility.py

from dataclasses import replace
from typing import Tuple

from src.surfaces.definition import PatternDefinition, SurfaceDefinition, ObservationSpec
from src.capabilities import CapabilityRef


class SurfaceCompatibilityError(Exception):
    """Surface 兼容性错误"""
    pass


# matcher 字符串到 Capability ID 的映射
LEGACY_MATCHER_TO_CAPABILITY = {
    "keyword": "builtin.keyword",
    "regex": "builtin.regex",
    "quotation": "builtin.quotation",
}


def _matcher_to_ref(matcher: str) -> CapabilityRef:
    """将旧 matcher 字符串转换为 CapabilityRef"""
    # 如果已经包含 "."，假设它已经是 capability id
    if "." in matcher:
        return CapabilityRef.parse(matcher)

    try:
        capability_id = LEGACY_MATCHER_TO_CAPABILITY[matcher]
    except KeyError:
        raise SurfaceCompatibilityError(
            f"Unknown legacy matcher: '{matcher}'. "
            f"Supported: {list(LEGACY_MATCHER_TO_CAPABILITY.keys())}"
        )
    return CapabilityRef.parse(capability_id)


def upgrade_pattern(pattern: PatternDefinition) -> PatternDefinition:
    """
    升级 PatternDefinition：matcher → capability_ref
    
    幂等性：多次调用结果相同
    """
    # 如果已有 capability_ref，保留（幂等）
    if pattern.capability_ref is not None:
        return pattern

    # 如果没有 matcher，无法升级
    if pattern.matcher is None:
        return pattern

    ref = _matcher_to_ref(pattern.matcher)
    return replace(
        pattern,
        capability_ref=ref,
        matcher=None,  # 清除旧字段，防止 Runtime 继续依赖
    )


def upgrade_observation(obs: ObservationSpec) -> ObservationSpec:
    """升级 ObservationSpec 中的所有 Pattern"""
    upgraded = tuple(upgrade_pattern(p) for p in obs.patterns)
    return replace(obs, patterns=upgraded)


def upgrade_surface(surface: SurfaceDefinition) -> SurfaceDefinition:
    """升级整个 SurfaceDefinition"""
    upgraded_obs = upgrade_observation(surface.observation)
    return replace(surface, observation=upgraded_obs)


def upgrade_surfaces(surfaces: Tuple[SurfaceDefinition, ...]) -> Tuple[SurfaceDefinition, ...]:
    """升级多个 SurfaceDefinition（Loader 唯一入口）"""
    return tuple(upgrade_surface(s) for s in surfaces)


# 幂等性检查辅助
def is_upgraded(pattern: PatternDefinition) -> bool:
    """检查 Pattern 是否已升级"""
    return pattern.capability_ref is not None and pattern.matcher is None