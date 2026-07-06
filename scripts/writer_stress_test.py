#!/usr/bin/env python
"""
Writer Stress Test - 直接调用 LLM，不依赖 AI Factory 框架
"""

import asyncio
import json
import httpx
from pathlib import Path
from datetime import datetime
from typing import Dict

from openai import AsyncOpenAI


# ============ 测试场景定义 ============

TEST_SCENES = {
    "S1": {
        "name": "进入剑阁",
        "scene_plan": {
            "goal": "进入剑阁探索",
            "conflict": "禁制阻挡",
            "outcome": "成功进入",
            "characters": ["林逸"],
            "must_events": ["触发禁制", "破解禁制"],
        },
    },
    "S2": {
        "name": "灵田争夺",
        "scene_plan": {
            "goal": "获得灵田分配权",
            "conflict": "与赵虎争夺",
            "outcome": "赢得分配权",
            "characters": ["林逸", "赵虎", "灵田管事"],
            "must_events": ["争夺灵田分配权", "首次同门冲突"],
        },
    },
    "S3": {
        "name": "学习御剑术",
        "scene_plan": {
            "goal": "掌握基础御剑术",
            "conflict": "灵力控制不稳定",
            "outcome": "完成基础飞行",
            "characters": ["林逸", "御剑术长老"],
            "must_events": ["学习御剑术"],
        },
    },
}


def build_prompt(scene: Dict, prompt_type: str) -> str:
    plan = scene["scene_plan"]
    
    base = f"""根据以下场景计划，生成一段小说正文（300-500字）。

**场景目标**：{plan['goal']}
**核心冲突**：{plan['conflict']}
**预期结果**：{plan['outcome']}
**参与角色**：{', '.join(plan['characters'])}
**必须发生的事件**：{', '.join(plan['must_events'])}"""

    if prompt_type == "baseline":
        return base + "\n\n请生成正文。"
    
    elif prompt_type == "conflict":
        return base + """

**🔴 戏剧性要求（必须遵守）**：

1. 主角必须有明确想要的东西
2. 必须有明确的阻碍（有人阻止主角）
3. 主角必须做选择（至少一次两难抉择）
4. 选择必须有代价
5. 至少有 2 个角色的对话或对抗

请生成符合以上要求的正文。"""
    
    else:  # structured
        drama = {
            "S1": {
                "goal": "林逸想进入剑阁",
                "obstacle": "剑阁管事拒绝放行",
                "pressure": "半个时辰后剑阁将关闭",
                "decision": "违抗命令强行进入",
                "cost": "得罪管事，日后被穿小鞋",
            },
            "S2": {
                "goal": "林逸想获得灵田分配权",
                "obstacle": "赵虎抢先占位",
                "pressure": "灵田分配每月仅一次",
                "decision": "公开冲突",
                "cost": "赢得灵田但结下仇怨",
            },
            "S3": {
                "goal": "林逸想学会御剑术",
                "obstacle": "灵力控制不稳定，多次失败",
                "pressure": "三日后有入门考核",
                "decision": "继续苦练直到成功",
                "cost": "消耗过多灵力",
            },
        }
        d = drama.get(scene.get("_id", "S1"), drama["S1"])
        
        return base + f"""

**🔴 戏剧结构（必须严格按照以下结构执行）**：

- 主角想要：{d['goal']}
- 阻碍：{d['obstacle']}
- 压力：{d['pressure']}
- 决定：{d['decision']}
- 代价：{d['cost']}

请将以上结构扩展为正文，必须包含：欲望、阻碍、压力、选择、代价、角色互动。"""


async def call_llm(prompt: str) -> str:
    # 明确禁用代理，避免 socks5 错误
    transport = httpx.AsyncHTTPTransport(proxy=None)
    async with httpx.AsyncClient(transport=transport) as client:
        openai_client = AsyncOpenAI(
            api_key="not-needed",
            base_url="http://localhost:8082",
            http_client=client,
        )
        response = await openai_client.chat.completions.create(
            model="Qwen3-32B-Q5_K_M-writer",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""

async def main():
    print("\n" + "=" * 70)
    print("🧪 Writer Stress Test (Direct)")
    print("=" * 70 + "\n")

    results = {}

    for scene_id, scene in TEST_SCENES.items():
        scene["_id"] = scene_id
        print(f"📌 场景: {scene['name']} ({scene_id})")
        results[scene_id] = {}

        for prompt_type in ["baseline", "conflict", "structured"]:
            print(f"  ▶ {prompt_type}...", end="", flush=True)
            prompt = build_prompt(scene, prompt_type)
            
            try:
                text = await call_llm(prompt)
                results[scene_id][prompt_type] = {
                    "text": text,
                    "length": len(text),
                }
                print(f" ✅ {len(text)} 字符")
            except Exception as e:
                print(f" ❌ 失败: {e}")
                results[scene_id][prompt_type] = {"error": str(e)}

        print()

    # 保存结果
    output_dir = Path("./stress_test_results_direct")
    output_dir.mkdir(exist_ok=True)
    
    for scene_id, data in results.items():
        with open(output_dir / f"{scene_id}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 结果保存到 {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())