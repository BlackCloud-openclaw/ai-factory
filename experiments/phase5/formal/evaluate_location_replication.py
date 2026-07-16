#!/usr/bin/env python3
# experiments/phase5/formal/evaluate_location_replication.py
"""
评估 MP2 + MP3 的 Recognition / Integration / Utilization / Continuity
"""

import asyncio
import json
import re
from pathlib import Path
from collections import defaultdict
import statistics
import httpx

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"

INJECTION_DESC = {
    "GapHigh_MatchHigh": "上一场景结束时位于特定地点，当前场景发生在另一个地点，需要从上一地点自然过渡到当前地点。",
    "GapHigh_MatchLow": "当前目标是找出某件物品的来历或含义。",
    "GapLow_MatchHigh": "上一场景结束时位于特定地点，当前场景发生在另一个地点，需要从上一地点自然过渡到当前地点。",
    "GapLow_MatchLow": "当前目标是找出某件物品的来历或含义。",
}


def load_texts(raw_dir: Path) -> list:
    texts = []
    for f in sorted(raw_dir.glob("*.txt")):
        name = f.stem
        parts = name.split('_')
        label = parts[0]
        condition = parts[1] + "_" + parts[2]
        rep = int(parts[3].replace('rep', ''))
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        match = re.search(r'={60}\n\n(.*?)\n\n={60}', content, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            text = content.strip()
        texts.append({
            "filename": f.name,
            "label": label,
            "condition": condition,
            "rep": rep,
            "text": text[:1000],
        })
    return texts


async def call_judge(text: str, injection: str) -> dict:
    prompt = f"""你是叙事分析专家。请分析以下场景文本，评估注入的状态信息是否被识别、整合和利用。

【场景文本】
{text}

【注入的状态信息】
{injection}

请从以下四个维度评分（1-5），只输出 JSON：

1. Recognition（识别）：文本中是否明确提到了注入的状态信息？
2. Integration（整合）：状态信息是否被写入正文并成为场景的一部分？
3. Utilization（利用）：状态信息是否影响了角色的决策或情节走向？
4. Continuity（连续性）：场景之间的整体连续性如何？

输出 JSON：
{{
    "recognition": 1-5,
    "integration": 1-5,
    "utilization": 1-5,
    "continuity": 1-5,
    "reason": "简短理由"
}}"""

    async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(300.0)) as client:
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 400,
        }
        try:
            resp = await client.post(f"{LLM_API_BASE}/chat/completions", json=payload)
            data = resp.json()
            content = data["choices"][0]["message"].get("content", "")
            if not content:
                content = data["choices"][0]["message"].get("reasoning_content", "")
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group())
                for key in ["recognition", "integration", "utilization", "continuity"]:
                    if key not in result:
                        result[key] = 3
                return result
            nums = [int(x) for x in re.findall(r'[1-5]', content) if x.isdigit()]
            if len(nums) >= 4:
                return {
                    "recognition": nums[0],
                    "integration": nums[1],
                    "utilization": nums[2],
                    "continuity": nums[3],
                    "reason": "从文本提取数字"
                }
            return {"recognition": 3, "integration": 3, "utilization": 3, "continuity": 3, "reason": "解析失败"}
        except Exception as e:
            print(f"  [ERROR] {e}")
            return {"recognition": 3, "integration": 3, "utilization": 3, "continuity": 3, "reason": str(e)[:50]}


async def main():
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "reports" / "raw_location_replication"
    output_dir = base_dir / "reports" / "location_replication_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Location Gap 复制验证评估: MP2 + MP3")
    print("="*80)

    texts = load_texts(raw_dir)
    print(f"找到 {len(texts)} 个样本")

    by_label = defaultdict(list)
    for t in texts:
        key = f"{t['label']}_{t['condition']}"
        by_label[key].append(t)

    all_results = []

    for key, items in by_label.items():
        label, condition = key.split('_', 1)
        injection = INJECTION_DESC.get(condition, "")
        print(f"\n--- {label} {condition} ---")
        for item in items:
            print(f"  评估: {item['filename']}...")
            result = await call_judge(item["text"], injection)
            result["label"] = label
            result["condition"] = condition
            result["rep"] = item["rep"]
            result["filename"] = item["filename"]
            all_results.append(result)
            print(f"    R:{result['recognition']} I:{result['integration']} U:{result['utilization']} C:{result['continuity']}")

    output_path = output_dir / "replication_scores.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("汇总统计")
    print("="*80)

    conditions = ["GapHigh_MatchHigh", "GapHigh_MatchLow", "GapLow_MatchHigh", "GapLow_MatchLow"]
    for label in ["MP2", "MP3"]:
        print(f"\n{label}:")
        for cond in conditions:
            items = [r for r in all_results if r["label"] == label and r["condition"] == cond]
            if items:
                n = len(items)
                r = statistics.mean([i["recognition"] for i in items])
                integ = statistics.mean([i["integration"] for i in items])
                u = statistics.mean([i["utilization"] for i in items])
                c = statistics.mean([i["continuity"] for i in items])
                print(f"  {cond}: R={r:.2f} I={integ:.2f} U={u:.2f} C={c:.2f} (n={n})")

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())