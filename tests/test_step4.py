#!/usr/bin/env python
"""测试 Step 4: Voiceprint 和 ContextCompiler"""
import sys
sys.path.insert(0, '/home/data/projects/ai_factory')

from src.writing import WorldState, CharacterState, Realm
from src.writing.voiceprint import VoiceprintRegistry
from src.writing.context_compiler import ContextCompiler

def test_voiceprint():
    registry = VoiceprintRegistry("config/voiceprints.yaml")
    constraint = registry.build_prompt_constraint("林逸")
    print("林逸的语言约束：")
    print(constraint)
    print()

def test_context_compiler():
    # 创建测试状态
    lin_yi = CharacterState(
        name="林逸",
        realm=Realm.REFINING_QI,
        realm_level=3,
        hp=85,
        mp=70,
        inventory=["玉佩", "丹药"],
        location="青云宗"
    )
    world = WorldState(
        characters={"林逸": lin_yi},
        global_flags={"玉佩觉醒": True, "主线进度": 2}
    )
    
    compiler = ContextCompiler(max_tokens=500)
    compiled = compiler.compile(world)
    print("编译后的上下文：")
    print(compiled)
    print()
    
    # 测试 Planner 专用编译
    outline = {"title": "修仙传", "volumes": [{"volume_num": 1, "title": "初入修仙界"}]}
    planner_ctx = compiler.compile_for_planner(world, 1, 3, outline)
    print("Planner 上下文：")
    print(planner_ctx)

if __name__ == "__main__":
    test_voiceprint()
    test_context_compiler()
    print("✅ Step 4 测试通过")