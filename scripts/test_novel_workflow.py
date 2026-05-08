import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.orchestrator.graph import compile_workflow
from src.orchestrator.state import AgentState

async def main():
    state = AgentState(
        user_input="生成测试场景",
        task_type="scene_plan",
        validation_mode="novel",  # 新增
        scene_plan={
            "goal": "林风遇到神秘老人",
            "conflict": "老人不信任林风",
            "outcome": "林风通过考验",
            "characters": ["林风", "神秘老人"]
        },
        writing_constraints={
            "character_states": {"林风": {"realm": "炼气", "mood": "坚定"}},
            "must_events": ["老人考验林风"],
            "forbidden_events": ["林风死亡"],
            "style_profile": {"tone": "热血"}
        },
        current_volume=1,
        current_chapter=1,
        current_scene=1,
        outline={}
    )
    
    workflow = compile_workflow()
    result = await workflow.ainvoke(state.dict())
    
    print("生成的场景正文：")
    print(result.get("scene_text", "未生成文本"))
    print("\n最终状态：", result.get("current_node"))

if __name__ == "__main__":
    asyncio.run(main())