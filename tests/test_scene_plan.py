import asyncio
import json
import asyncpg
from src.orchestrator.graph import compile_workflow
from src.orchestrator.state import AgentState
from src.config import config

async def main():
    # 1. 从数据库加载大纲
    conn = await asyncpg.connect(config.postgres_dsn)
    row = await conn.fetchrow("SELECT outline FROM novels WHERE novel_id='test_debug'")
    if not row:
        print("No outline found")
        return
    outline = json.loads(row["outline"])
    await conn.close()

    # 2. 构建状态，触发场景计划生成
    state = AgentState(
        user_input="根据大纲生成第一章的第一个场景计划",
        task_type="scene_plan",
        novel_id="test_debug",
        outline=outline,
        current_volume=1,
        current_chapter=1,
        current_scene=1,
    )

    workflow = compile_workflow()
    result = await workflow.ainvoke(state.dict())
    
    scene_plan = result.get("scene_plan")
    print("生成场景计划：", json.dumps(scene_plan, indent=2, ensure_ascii=False))

    # 3. 紧接着生成正文（需要将场景计划转为 writing_constraints）
    if scene_plan:
        writing_state = AgentState(
            user_input="根据场景计划写出场景正文",
            task_type="scene_plan",  # 沿用，但会走 writer 分支
            novel_id="test_debug",
            outline=outline,
            scene_plan=scene_plan[0] if isinstance(scene_plan, list) else scene_plan,
            writing_constraints={
                "character_states": {"林风": {"realm": "炼气", "level": 1}},
                "must_events": ["林风触发残阵"],
                "forbidden_events": ["林风死亡"],
                "style_profile": {"tone": "热血"}
            }
        )
        result2 = await workflow.ainvoke(writing_state.dict())
        print("\n生成的场景正文：")
        print(result2.get("scene_text", ""))

if __name__ == "__main__":
    asyncio.run(main())