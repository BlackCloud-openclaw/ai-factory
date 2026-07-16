# experiments/phase5/formal/phase5_4_experiment.py
#!/usr/bin/env python3
"""
Phase 5.4: Cross-Scene Generalization
伦理型两难场景 × 3 State 条件 × 5 次生成
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

OUTPUT_DIR = Path(__file__).parent / "reports" / "phase5_4"
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

CONDITIONS = {
    "control": {
        "name": "Control（无 State）",
        "state": "",
        "description": "请从以上场景继续，写一段林逸做出决定的场景。"
    },
    "task": {
        "name": "Task State",
        "state": "宗门三日内有重要任务需要林逸参与，必须在规定时间内返回。",
        "description": "请从以上场景继续，写一段林逸做出决定的场景。林逸知道宗门有重要任务等待他。"
    },
    "exploration": {
        "name": "Exploration State",
        "state": "禁地方向的灵力波动与林逸一直在追查的某个秘密高度相关。",
        "description": "请从以上场景继续，写一段林逸做出决定的场景。林逸知道禁地的波动与追查的秘密有关。"
    }
}

WRITER_PROMPT_TEMPLATE = """你是一位小说作者。请根据以下场景开头，续写一段场景正文（300-500字）。

【场景开头】
{scene_a}

【续写要求】
{description}

{state}

写作要求：
1. 保持与开头一致的第三人称叙述风格
2. 描写林逸的思考过程、决策心理和行动
3. 自然过渡到后续情节
4. 不要添加任何解释或元评论
5. 只输出续写的正文，不要包含任何额外内容"""

# ============================================================
# 实验执行
# ============================================================

class Phase54Experiment:
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
    
    async def run_condition(self, cond_key: str, samples: int = 5):
        cond = CONDITIONS[cond_key]
        print(f"\n运行: {cond['name']}")
        
        results = []
        for s in range(samples):
            print(f"  样本 {s+1}/{samples}...")
            state_text = f"\n【状态信息】\n{cond['state']}" if cond['state'] else ""
            prompt = WRITER_PROMPT_TEMPLATE.format(
                scene_a=SCENE_A,
                description=cond['description'],
                state=state_text
            )
            response = await self._call_llm(prompt)
            results.append({
                "sample": s,
                "condition": cond_key,
                "condition_name": cond['name'],
                "state": cond['state'],
                "text": response,
                "length": len(response)
            })
            await asyncio.sleep(2.0)
        return results
    
    async def run_all(self):
        print("=" * 60)
        print("Phase 5.4: Cross-Scene Generalization")
        print("=" * 60)
        print(f"场景: 伦理型两难")
        print(f"条件: Control, Task State, Exploration State")
        print(f"每个条件: 5 次生成")
        print("=" * 60)
        
        all_results = []
        for cond_key in ["control", "task", "exploration"]:
            results = await self.run_condition(cond_key)
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
        lines.append("# Phase 5.4 实验报告")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().isoformat()}")
        lines.append(f"**模型**: {LLM_MODEL}")
        lines.append(f"**场景**: 伦理型两难")
        lines.append("")
        
        lines.append("## 长度统计")
        lines.append("| 条件 | 平均长度 | 最小 | 最大 |")
        lines.append("|------|----------|------|------|")
        for cond_key in ["control", "task", "exploration"]:
            cond_results = [r for r in results if r["condition"] == cond_key]
            if cond_results:
                lengths = [r["length"] for r in cond_results]
                avg = sum(lengths) / len(lengths)
                lines.append(f"| {CONDITIONS[cond_key]['name']} | {avg:.0f} | {min(lengths)} | {max(lengths)} |")
        lines.append("")
        
        # 样本预览
        lines.append("## 样本预览")
        for cond_key in ["control", "task", "exploration"]:
            cond_results = [r for r in results if r["condition"] == cond_key]
            if cond_results:
                lines.append(f"\n### {CONDITIONS[cond_key]['name']}")
                for i, r in enumerate(cond_results[:2]):
                    text = r["text"][:400] + "..." if len(r["text"]) > 400 else r["text"]
                    lines.append(f"\n**样本 {i+1}** (长度: {r['length']}):")
                    lines.append(f"\n{text}")
        
        return "\n".join(lines)


async def main():
    exp = Phase54Experiment()
    await exp.run_all()

if __name__ == "__main__":
    asyncio.run(main())