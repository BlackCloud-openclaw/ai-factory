# src/writing/state_change_factory.py
"""
StateChange Factory - Phase 13.2.3A

根据 EventType 和上下文生成 StateChange 对象。
确保生成的每个 StateChange 都有稳定的 ID 和来源标记。
"""

import hashlib
import re
from typing import Optional, Dict, Any

from src.writing.planning_contract import StateChange, SignalSource
from src.writing.event_classifier import EventType


class StateChangeFactory:
    """
    StateChange 工厂。

    职责：
    1. 根据 EventType 创建 StateChange
    2. 生成稳定 ID（基于 contract_id + event_type + 归一化内容）
    3. 标记 source = INFERRED（工厂生成的所有信号均为推断）
    """

    # 默认 Actor（当上下文中无法提取时）
    DEFAULT_ACTOR = "林逸"

    @classmethod
    def create(
        cls,
        event_type: EventType,
        context: Dict[str, Any],
        contract_id: str,
    ) -> Optional[StateChange]:
        """
        根据 EventType 创建 StateChange。

        Args:
            event_type: 事件类型
            context: 上下文，包含：
                - text: 原始 must_event 文本
                - actor: 可选，角色名
                - item: 可选，物品名
                - location: 可选，地点名
                - from_char: 可选，关系发起方
                - to_char: 可选，关系接收方
            contract_id: 契约 ID（用于生成稳定 ID）

        Returns:
            StateChange 或 None（如果上下文不足）
        """
        if event_type == EventType.REALM_ADVANCE:
            return cls._create_realm_advance(context, contract_id)
        elif event_type == EventType.ITEM_ACQUIRE:
            return cls._create_item_acquire(context, contract_id)
        elif event_type == EventType.ITEM_LOST:
            return cls._create_item_lost(context, contract_id)
        elif event_type == EventType.LOCATION_CHANGE:
            return cls._create_location_change(context, contract_id)
        elif event_type == EventType.RELATION_CHANGE:
            return cls._create_relation_change(context, contract_id)
        elif event_type == EventType.PLOT_REVEAL:
            return cls._create_plot_reveal(context, contract_id)
        else:
            return None

    @classmethod
    def _generate_id(cls, contract_id: str, event_type: str, text: str) -> str:
        """生成稳定 ID。"""
        # 归一化文本：去停用词、去标点
        clean = re.sub(r'[\s，。、！？；：""''（）]', '', text)
        raw = f"{contract_id}|{event_type}|{clean}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def _extract_actor(cls, context: Dict) -> str:
        """从上下文提取 actor。"""
        actor = context.get("actor")
        if actor:
            return actor
        text = context.get("text", "")
        match = re.match(r'^([\u4e00-\u9fff]{2,4})', text)
        if match:
            return match.group(1)
        return cls.DEFAULT_ACTOR

    @classmethod
    def _extract_item(cls, context: Dict) -> str:
        """从上下文提取物品名。"""
        item = context.get("item")
        if item:
            return item
        text = context.get("text", "")
        match = re.search(r'(?:获得|捡到|得到|夺取|缴获|拾取)[\s]*([\u4e00-\u9fff]{2,8})', text)
        if match:
            return match.group(1)
        return "未知物品"

    @classmethod
    def _extract_location(cls, context: Dict) -> str:
        """从上下文提取地点名。"""
        location = context.get("location")
        if location:
            return location
        text = context.get("text", "")
        match = re.search(r'(?:进入|踏入|抵达|来到|离开)[\s]*([\u4e00-\u9fff]{2,8})', text)
        if match:
            return match.group(1)
        return "未知地点"

    @classmethod
    def _extract_relationship(cls, context: Dict) -> tuple:
        """从上下文提取关系双方。"""
        from_char = context.get("from_char")
        to_char = context.get("to_char")
        if from_char and to_char:
            return from_char, to_char

        text = context.get("text", "")
        match = re.search(r'与[\s]*([\u4e00-\u9fff]{2,4})', text)
        to_char = match.group(1) if match else "unknown"
        from_char = cls._extract_actor(context)

        return from_char, to_char

    @classmethod
    def _create_state_change(cls, sc_type: str, sc_id: str, **kwargs) -> StateChange:
        """创建带 source=INFERRED 的 StateChange。"""
        # ========== 确保 source 为枚举 ==========
        return StateChange(
            id=sc_id,
            type=sc_type,
            source=SignalSource.INFERRED,   # ✅ 枚举
            **kwargs,
        )

    @classmethod
    def _create_realm_advance(cls, context: Dict, contract_id: str) -> Optional[StateChange]:
        text = context.get("text", "")
        actor = cls._extract_actor(context)

        realm_match = re.search(r'([\u4e00-\u9fff]{2,4})(?:境界|境|期)', text)
        to_realm = realm_match.group(1) if realm_match else "金丹"

        level_match = re.search(r'(\d+)[层|重]', text)
        to_level = int(level_match.group(1)) if level_match else 1

        sc_id = cls._generate_id(contract_id, "realm", text)
        return cls._create_state_change(
            "realm", sc_id,
            actor=actor,
            to_major_realm=to_realm,
            to_minor_stage=to_level,
        )

    @classmethod
    def _create_item_acquire(cls, context: Dict, contract_id: str) -> Optional[StateChange]:
        text = context.get("text", "")
        actor = cls._extract_actor(context)
        item = cls._extract_item(context)

        sc_id = cls._generate_id(contract_id, "inventory_acquire", text)
        return cls._create_state_change(
            "inventory", sc_id,
            actor=actor,
            item=item,
            operation="acquire",
            quantity=1,
        )

    @classmethod
    def _create_item_lost(cls, context: Dict, contract_id: str) -> Optional[StateChange]:
        text = context.get("text", "")
        actor = cls._extract_actor(context)
        item = cls._extract_item(context)

        sc_id = cls._generate_id(contract_id, "inventory_lost", text)
        return cls._create_state_change(
            "inventory", sc_id,
            actor=actor,
            item=item,
            operation="lose",
            quantity=1,
        )

    @classmethod
    def _create_location_change(cls, context: Dict, contract_id: str) -> Optional[StateChange]:
        text = context.get("text", "")
        actor = cls._extract_actor(context)
        location = cls._extract_location(context)

        sc_id = cls._generate_id(contract_id, "location", text)
        return cls._create_state_change(
            "location", sc_id,
            actor=actor,
            location=location,
        )

    @classmethod
    def _create_relation_change(cls, context: Dict, contract_id: str) -> Optional[StateChange]:
        text = context.get("text", "")
        from_char, to_char = cls._extract_relationship(context)

        if any(kw in text for kw in ["交恶", "决裂", "结仇", "冲突"]):
            delta = -20
        elif any(kw in text for kw in ["结盟", "和解", "亲近"]):
            delta = 20
        else:
            delta = -10

        sc_id = cls._generate_id(contract_id, "relationship", text)
        return cls._create_state_change(
            "relationship", sc_id,
            from_char=from_char,
            to_char=to_char,
            delta=delta,
        )

    @classmethod
    def _create_plot_reveal(cls, context: Dict, contract_id: str) -> Optional[StateChange]:
        text = context.get("text", "")
        # 提取 flag 名称
        flag = re.sub(r'[\s，。、！？；：""''（）]', '_', text[:30])
        flag = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff]', '_', flag)

        sc_id = cls._generate_id(contract_id, "plot_flag", text)
        return cls._create_state_change(
            "plot_flag", sc_id,
            name=flag,
            value=True,
        )