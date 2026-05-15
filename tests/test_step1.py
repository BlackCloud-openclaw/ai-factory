#!/usr/bin/env python
"""测试 Step 1 的模块"""
import sys
sys.path.insert(0, '/home/data/projects/ai_factory')

from src.writing.world_state import WorldState, CharacterState, Realm
from src.writing.events import RealmUpgradeEvent, ItemAcquireEvent
from src.writing.delta import StateDelta

def test_world_state():
    # 创建角色
    lin_yi = CharacterState(
        name="林逸",
        realm=Realm.REFINING_QI,
        realm_level=3,
        inventory=["玉佩", "丹药"]
    )
    
    # 创建世界状态
    world = WorldState(
        characters={"林逸": lin_yi},
        revision=0
    )
    
    print(f"角色: {world.get_character('林逸').full_realm()}")
    print(f"活跃角色: {world.get_active_characters()}")
    print(f"WorldState JSON 长度: {len(world.model_dump_json())}")
    
    return world

def test_delta():
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
        )
    ])
    
    print(f"Delta 包含 {len(delta.events)} 个事件")
    print(f"Prompt 友好格式: {delta.to_prompt_friendly()}")
    
    return delta

if __name__ == "__main__":
    print("=" * 50)
    print("测试 WorldState")
    print("=" * 50)
    test_world_state()
    
    print("\n" + "=" * 50)
    print("测试 StateDelta")
    print("=" * 50)
    test_delta()
    
    print("\n✅ Step 1 模块测试通过")