# experiments/phase6/phase6_1_experiment.py
#!/usr/bin/env python3
"""
Phase 6.1: Propagation Policy Ablation + Baseline

四组实验：
- Baseline（无 Runtime Policy）
- Conservative（DISABLED + ENHANCED）
- Adaptive（ASSIST + ENHANCED）
- Aggressive（PRIMARY + ENHANCED）

每组合计 5 次生成，共 20 篇样本。
"""

import asyncio
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import httpx

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"

OUTPUT_DIR = Path(__file__).parent / "reports" / "phase6_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 场景材料
# ============================================================

SCENE_A = """林逸站在岔路口，左边的路通向宗门，右边的路通向禁地。师父交代的任务是三天内返回宗门复命，但禁地方向隐约传来异样的灵力波动。他停下脚步，望着两条路。"""

OPTIONS = [
    "立即返回宗门，遵守师命",
    "立即前往禁地，探查灵力波动",
    "立即在原地留下标记，等待同伴",
    "立即登上高处，观察禁地方向再决定"
]

STATE = "禁地方向的灵力波动与林逸一直在追查的某个秘密高度相关。"

# ============================================================
# Policy 配置
# ============================================================

POLICIES = {
    "baseline": {
        "name": "Baseline（无 Runtime）",
        "prediction_policy": None,
        "realization_policy": None,
        "description": "请从以上场景继续，写一段林逸做出决定的场景。",
        "constraints": ""
    },
    "conservative": {
        "name": "Conservative（DISABLED + ENHANCED）",
        "prediction_policy": "DISABLED",
        "realization_policy": "ENHANCED",
        "description": "请从以上场景继续，写一段林逸做出决定的场景。",
        "constraints": """
【Runtime 约束：保守策略】
- 不尝试改变事件选择（Prediction DISABLED）：场景的事件走向已锁定，遵循默认的下一事件。
- 增强注入 State（Realization ENHANCED）：State 信息应在叙事实现中占据重要位置。
"""
    },
    "adaptive": {
        "name": "Adaptive（ASSIST + ENHANCED）",
        "prediction_policy": "ASSIST",
        "realization_policy": "ENHANCED",
        "description": "请从以上场景继续，写一段林逸做出决定的场景。",
        "constraints": """
【Runtime 约束：自适应策略】
- State 作为参考（Prediction ASSIST）：State 信息可作为辅助参考，但不强制改变事件选择。
- 增强注入 State（Realization ENHANCED）：State 信息应在叙事实现中占据重要位置。
"""
    },
    "aggressive": {
        "name": "Aggressive（PRIMARY + ENHANCED）",
        "prediction_policy": "PRIMARY",
        "realization_policy": "ENHANCED",
        "description": "请从以上场景继续，写一段林逸做出决定的场景。",
        "constraints": """
【Runtime 约束：激进策略】
- State 主导事件选择（Prediction PRIMARY）：State 信息应作为决定下一事件的主要依据。
- 增强注入 State（Realization ENHANCED）：State 信息应在叙事实现中占据重要位置。
"""
    }
}

WRITER_PROMPT_TEMPLATE = """你是一位小说作者。请根据以下场景开头，续写一段场景正文（300-500字）。

【场景开头】
{scene_a}

【续写要求】
{description}

【状态信息】
{state}

{constraints}

写作要求：
1. 保持与开头一致的第三人称叙述风格
2. 必须体现角色形成决策的思考过程
3. 结尾必须出现明确的行动倾向（最终决定或明显偏向）
4. 不要添加任何解释或元评论
5. 只输出续写的正文，不要包含任何额外内容"""

# ============================================================
# 实验执行
# ============================================================

class Phase61Experiment:
    def __init__(self):
        self.results = []
    
    async def _call_llm(self, prompt: str, retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(
                    trust_env=False,
                    timeout=httpx.Timeout(120.0, connect=10.0)
                ) as client:
                    payload = {
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 2048,
                    }
                    response = await client.post(
                        f"{LLM_API_BASE}/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"].get("content", "")
                    if content and content.strip():
                        return content
                    print(f"  [WARNING] Empty response, attempt {attempt+1}/{retries}")
            except Exception as e:
                print(f"  [ERROR] Attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep((attempt + 1) * 2)
        return ""
    
    async def run_policy(self, policy_key: str, samples: int = 5):
        policy = POLICIES[policy_key]
        print(f"\n运行: {policy['name']}")
        
        results = []
        for s in range(samples):
            print(f"  样本 {s+1}/{samples}...")
            prompt = WRITER_PROMPT_TEMPLATE.format(
                scene_a=SCENE_A,
                description=policy['description'],
                state=STATE,
                constraints=policy['constraints']
            )
            response = await self._call_llm(prompt)
            results.append({
                "sample": s,
                "policy": policy_key,
                "policy_name": policy['name'],
                "prediction_policy": policy['prediction_policy'],
                "realization_policy": policy['realization_policy'],
                "text": response,
                "length": len(response)
            })
            await asyncio.sleep(2.0)
        return results
    
    async def run_all(self):
        print("=" * 60)
        print("Phase 6.1: Propagation Policy Ablation")
        print("=" * 60)
        print(f"场景: 伦理型两难")
        print(f"State: {STATE[:40]}...")
        print(f"策略: Baseline, Conservative, Adaptive, Aggressive")
        print(f"每个策略: 5 次生成")
        print("=" * 60)
        
        all_results = []
        for policy_key in ["baseline", "conservative", "adaptive", "aggressive"]:
            results = await self.run_policy(policy_key)
            all_results.extend(results)
        
        # 保存原始数据
        raw_path = OUTPUT_DIR / "raw_data.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n原始数据保存到: {raw_path}")
        
        # 生成简要报告
        report_path = OUTPUT_DIR / "report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.generate_report(all_results))
        print(f"报告保存到: {report_path}")
        
        return all_results
    
    def generate_report(self, results: list) -> str:
        lines = []
        lines.append("# Phase 6.1 实验报告")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().isoformat()}")
        lines.append(f"**模型**: {LLM_MODEL}")
        lines.append(f"**场景**: 伦理型两难")
        lines.append(f"**策略数**: 4（Baseline, Conservative, Adaptive, Aggressive）")
        lines.append(f"**每策略样本数**: 5")
        lines.append("")
        
        lines.append("## 长度统计")
        lines.append("| 策略 | 平均长度 | 最小 | 最大 |")
        lines.append("|------|----------|------|------|")
        for policy_key in ["baseline", "conservative", "adaptive", "aggressive"]:
            policy_results = [r for r in results if r["policy"] == policy_key]
            if policy_results:
                lengths = [r["length"] for r in policy_results]
                avg = sum(lengths) / len(lengths)
                lines.append(f"| {POLICIES[policy_key]['name']} | {avg:.0f} | {min(lengths)} | {max(lengths)} |")
        lines.append("")
        
        # 样本预览
        lines.append("## 样本预览")
        for policy_key in ["baseline", "conservative", "adaptive", "aggressive"]:
            policy_results = [r for r in results if r["policy"] == policy_key]
            if policy_results:
                lines.append(f"\n### {POLICIES[policy_key]['name']}")
                for i, r in enumerate(policy_results[:2]):
                    text = r["text"][:400] + "..." if len(r["text"]) > 400 else r["text"]
                    lines.append(f"\n**样本 {i+1}** (长度: {r['length']}):")
                    lines.append(f"\n{text}")
        
        lines.append("\n---\n")
        lines.append("## 编码框架（待人工编码）")
        lines.append("")
        lines.append("| 层级 | 名称 | 编码值 |")
        lines.append("|------|------|--------|")
        lines.append("| L0 | Control Fidelity | prediction_execution: 0-1, realization_execution: 0-1 |")
        lines.append("| L1 | Prediction | A/B/C/D + confidence: 1.0/0.7/0.5 |")
        lines.append("| L2 | Reasoning | Ignored / Mentioned / Integrated / Conflict / Dominant + 证据句 |")
        lines.append("| L3 | Justification | 多标签：Duty/Truth/Risk/Time/Emotion/Relationship/Curiosity/Other |")
        lines.append("| L4 | Construction | 从数据中提取 |")
        
        return "\n".join(lines)


async def main():
    exp = Phase61Experiment()
    await exp.run_all()

if __name__ == "__main__":
    asyncio.run(main())