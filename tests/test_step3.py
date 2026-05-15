#!/usr/bin/env python
"""测试 Step 3: apply_to 和事件应用逻辑"""
import sys
sys.path.insert(0, '/home/data/projects/ai_factory')

from src.writing.world_state import WorldState, CharacterState, Realm
from src.writing.events import (
    RealmUpgradeEvent, ItemAcquireEvent, RelationshipChangeEvent,
    HPChangedEvent, LocationEnterEvent, PlotFlagSetEvent
)
from src.writing.delta import StateDelta

def test_apply_to():
    # 创建初始状态
    lin_yi = CharacterState(
        name="林逸",
        realm=Realm.REFINING_QI,
        realm_level=3,
        hp=80,
        mp=60,
        inventory=["玉佩"],
        location="后山"
    )
    world = WorldState(
        characters={"林逸": lin_yi},
        revision=0
    )
    
    print("初始状态：")
    print(f"  境界: {lin_yi.full_realm()}")
    print(f"  HP: {lin_yi.hp}")
    print(f"  位置: {lin_yi.location}")
    print(f"  背包: {lin_yi.inventory}")
    
    # 创建 delta
    delta = StateDelta(events=[
        RealmUpgradeEvent(
            actor="林逸",
            from_realm="炼气",
            from_level=3,
            to_realm="炼气",
            to_level=4
        ),
        ItemAcquireEvent(
            actor="林逸",
            item="神秘古剑",
            source="古墓"
        ),
        HPChangedEvent(
            actor="林逸",
            delta=-15,
            new_hp=65
        ),
        LocationEnterEvent(
            actor="林逸",
            location="古墓密室",
            first_time=True
        ),
        PlotFlagSetEvent(
            flag="探索古墓",
            value=True
        )
    ])
    
    # 应用 delta
    new_world = delta.apply_to(world)
    new_lin_yi = new_world.characters["林逸"]
    
    print("\n应用 delta 后：")
    print(f"  境界: {new_lin_yi.full_realm()}")
    print(f"  HP: {new_lin_yi.hp}")
    print(f"  位置: {new_lin_yi.location}")
    print(f"  背包: {new_lin_yi.inventory}")
    print(f"  全局标记: {new_world.global_flags}")
    print(f"  版本号: {new_world.revision}")
    
    # 验证不可变性
    assert world.characters["林逸"].realm_level == 3, "原状态不应被修改"
    assert new_world.revision == 1, "版本号应增加"
    print("\n✅ 不可变性验证通过")
    
    return new_world

if __name__ == "__main__":
    test_apply_to()
    print("\n✅ Step 3 测试通过")