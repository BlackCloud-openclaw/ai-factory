# scripts/test_writer_capability.py
"""
对比不同 Capability 下 Writer 的输出
"""

import asyncio
from src.orchestrator.state import AgentState
from src.agents.writer import WritingAgent
from src.runtime import StateCapability, PredictionCapability, RealizationCapability, RetryStrategy


async def test_capability(capability, label):
    state = AgentState(
        novel_id='test',
        current_volume=1,
        current_chapter=1,
        current_scene_index=0,
        scene_plan={
            'scene_id': 'scene_reunion',
            'goal': '师兄重逢',
            'conflict': '十年恩怨',
            'characters': ['林逸', '师兄']
        },
        metadata={"transition_rigidity": 0.33},
        current_state={}
    )
    agent = WritingAgent()
    # 直接设置 capability（绕过 Router）
    agent._capability = capability
    result = await agent.run(state)
    return result.get('scene_text', '')


async def main():
    capabilities = [
        (StateCapability(
            prediction=PredictionCapability.PRIMARY,
            realization=RealizationCapability.ENHANCED,
            retry=RetryStrategy.FULL,
            reason="测试：Primary+Enhanced"
        ), "Primary_Enhanced"),
        (StateCapability(
            prediction=PredictionCapability.DISABLED,
            realization=RealizationCapability.NONE,
            retry=RetryStrategy.NONE,
            reason="测试：Disabled+None"
        ), "Disabled_None"),
    ]
    
    for cap, label in capabilities:
        print(f"\n{'='*60}")
        print(f"测试: {label}")
        print(f"{'='*60}")
        text = await test_capability(cap, label)
        print(f"输出长度: {len(text)}")
        print(text[:500] + "..." if len(text) > 500 else text)

if __name__ == "__main__":
    asyncio.run(main())