"""
XianxiaAdapter - 将现有 WorldState 映射到 KernelWorldState

仅做数据转换，不修改原状态。
"""

from typing import Dict, Any, List, Optional
import uuid

from src.writing.world_state import WorldState, CharacterState, Realm
from next.kernel.world import KernelWorldState
from next.kernel.entity import Entity, EntityType
from next.kernel.capability import Capability, CapabilityMode
from next.kernel.relation import Relation
from next.kernel.knowledge import Knowledge
from next.kernel.constraint import Constraint, ConstraintType


class XianxiaAdapter:
    @staticmethod
    def to_kernel(world: WorldState) -> KernelWorldState:
        kernel = KernelWorldState()

        # 1. 映射实体（角色）
        for name, char in world.characters.items():
            entity = Entity(
                id=name,
                name=name,
                type=EntityType.CHARACTER,
                attributes={
                    "location": char.location,
                    "hp": char.hp,
                    "mp": char.mp,
                    "beliefs": char.beliefs,
                    "attachments": char.attachments,
                    "self_image": char.self_image,
                    "moral_boundaries": char.moral_boundaries,
                }
            )
            kernel.entities[name] = entity

            # 2. 映射 Capability（境界、物品等）
            # 境界作为离散 Capability
            cultivation = Capability(
                name="cultivation",
                mode=CapabilityMode.DISCRETE,
                value=char.full_realm(),
                metadata={"realm_level": char.realm_level}
            )
            kernel.capabilities[f"{name}|cultivation"] = cultivation

            # 物品作为集合 Capability
            inventory = Capability(
                name="inventory",
                mode=CapabilityMode.SET,
                value=char.inventory.copy()
            )
            kernel.capabilities[f"{name}|inventory"] = inventory

        # 3. 映射关系（客观关系）
        for key, value in world.relationships.items():
            parts = key.split("|")
            if len(parts) == 2:
                relation = Relation(
                    id=f"rel_{parts[0]}_{parts[1]}",
                    from_entity=parts[0],
                    to_entity=parts[1],
                    relation_type="objective",
                    value=float(value),
                    confidence=1.0
                )
                kernel.relations[relation.id] = relation

        # 4. 映射认知关系（感知关系）
        for name, char in world.characters.items():
            for target, rel_info in char.perceived_relationships.items():
                rel_id = f"perception_{name}_{target}"
                relation = Relation(
                    id=rel_id,
                    from_entity=name,
                    to_entity=target,
                    relation_type="perceived",
                    value=float(rel_info.get("value", 0)),
                    confidence=rel_info.get("confidence", 0.0)
                )
                kernel.relations[rel_id] = relation

        # 5. 映射知识（从 beliefs 和认知关系中提取）
        for name, char in world.characters.items():
            for belief in char.beliefs:
                knowledge = Knowledge(
                    id=f"know_{name}_{belief[:20]}",
                    holder=name,
                    content=belief,
                    confidence=0.9,
                    source="belief"
                )
                kernel.knowledge[knowledge.id] = knowledge

        # 6. 映射约束（如果存在）
        if hasattr(world, 'constraints'):
            for c in world.constraints:
                if hasattr(c, 'to_dict'):
                    c_dict = c.to_dict()
                elif isinstance(c, dict):
                    c_dict = c
                else:
                    continue
                constraint = Constraint(
                    id=c_dict.get("id", str(uuid.uuid4())),
                    type=ConstraintType(c_dict.get("type", "rule")),
                    description=c_dict.get("description", ""),
                    owner=c_dict.get("owner", "world"),
                    target=c_dict.get("target"),
                    severity=c_dict.get("severity", 1.0),
                    is_active=c_dict.get("is_active", True),
                    expires_at=c_dict.get("expires_at")
                )
                kernel.constraints[constraint.id] = constraint

        # 7. 映射全局标记（作为世界 Capability）
        world_flags = Capability(
            name="global_flags",
            mode=CapabilityMode.SET,
            value=[k for k, v in world.global_flags.items() if v is True]
        )
        kernel.capabilities["world|global_flags"] = world_flags

        return kernel

    @staticmethod
    def get_coverage_report(world: WorldState, kernel: KernelWorldState) -> Dict[str, float]:
        """计算映射覆盖率"""
        total_fields = 0
        mapped_fields = 0

        # 角色字段覆盖率
        char_total = 0
        char_mapped = 0
        for name, char in world.characters.items():
            char_total += 1
            if name in kernel.entities:
                char_mapped += 1

            # 境界
            char_total += 1
            if f"{name}|cultivation" in kernel.capabilities:
                char_mapped += 1

            # 物品
            char_total += 1
            if f"{name}|inventory" in kernel.capabilities:
                char_mapped += 1

            # 信念
            char_total += len(char.beliefs)
            for belief in char.beliefs:
                found = any(k.content == belief for k in kernel.knowledge.values() if k.holder == name)
                if found:
                    char_mapped += 1

            # 认知关系
            char_total += len(char.perceived_relationships)
            for target in char.perceived_relationships:
                rel_id = f"perception_{name}_{target}"
                if rel_id in kernel.relations:
                    char_mapped += 1

        # 客观关系覆盖率
        rel_total = len(world.relationships)
        rel_mapped = sum(1 for r in kernel.relations.values() if r.relation_type == "objective")
        char_total += rel_total
        char_mapped += rel_mapped

        coverage = (char_mapped / char_total) if char_total > 0 else 0.0

        return {
            "total_fields": char_total,
            "mapped_fields": char_mapped,
            "coverage": coverage,
            "character_coverage": char_mapped / char_total if char_total > 0 else 0,
            "relationship_coverage": rel_mapped / rel_total if rel_total > 0 else 0,
        }