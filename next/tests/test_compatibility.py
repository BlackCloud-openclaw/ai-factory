import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db import init_db_pool, close_db_pool
from src.writing.world_state import WorldState
from src.writing.state_loader import load_state_at_chapter
from next.adapter.xianxia_adapter import XianxiaAdapter


async def test_compatibility():
    await init_db_pool()
    novel_id = "simple_long_novel_001"

    # 测试第 32 章
    world, _ = await load_state_at_chapter(novel_id, 1, 32)
    if world is None:
        print("❌ 无法加载世界状态")
        return

    kernel = XianxiaAdapter.to_kernel(world)
    report = XianxiaAdapter.get_coverage_report(world, kernel)

    print(f"📊 映射覆盖率报告:")
    print(f"   总字段数: {report['total_fields']}")
    print(f"   已映射字段数: {report['mapped_fields']}")
    print(f"   整体覆盖率: {report['coverage']:.1%}")
    print(f"   角色覆盖率: {report['character_coverage']:.1%}")
    print(f"   关系覆盖率: {report['relationship_coverage']:.1%}")

    # 关键实体检查
    assert "林逸" in kernel.entities, "主角未映射"
    assert "林逸|cultivation" in kernel.capabilities, "主角境界未映射"
    assert "林逸|inventory" in kernel.capabilities, "主角物品未映射"

    # 覆盖率门槛
    if report['coverage'] >= 0.8:
        print("✅ 覆盖率达标 (>=80%)")
    else:
        print(f"⚠️ 覆盖率 {report['coverage']:.1%} 低于 80%")

    await close_db_pool()


if __name__ == "__main__":
    asyncio.run(test_compatibility())