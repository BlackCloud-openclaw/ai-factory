#!/usr/bin/env python3
# experiments/phase5/formal/evaluate_mechanism_fixed.py
"""
机制评估 - 直接与 LLM Judge 交互
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

# 定义每个条件的注入描述（用于 Judge prompt）
INJECTION_DESC = {
    "A1": "上一场景结束时位于炼丹房，当前场景发生在静室，需要从炼丹房自然过渡到静室。",
    "B1": "当前目标是找出《太乙化神录》中的异常标记。",
    "C1": "上一场景结束时位于炼丹房，当前场景发生在静室，需要从炼丹房自然过渡到静室。",
    "D1": "当前目标是找出《太乙化神录》中的异常标记。",
}

# 条件映射
COND_MAP = {
    "A1": "GapHigh_MatchHigh",
    "B1": "GapHigh_MatchLow",
    "C1": "GapLow_MatchHigh",
    "D1": "GapLow_MatchLow",
}

def load_texts(raw_dir: Path) -> list:
    """加载所有生成的文本"""
    texts = []
    for f in sorted(raw_dir.glob("*.txt")):
        name = f.stem
        parts = name.split('_')
        # 格式: A1_GapHigh_MatchHigh_rep00_20260711_221022.txt
        label = parts[0]  # A1, B1, C1, D1
        rep_str = parts[3]  # rep00
        rep = int(rep_str.replace('rep', ''))
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
            "rep": rep,
            "text": text[:1000],
        })
    return texts


async def call_judge(text: str, injection: str) -> dict:
    """调用 Judge 进行四维评分"""
    prompt = f"""你是叙事分析专家。请分析以下场景文本，评估注入的状态信息是否被识别、整合和利用。

【场景文本】
{text}

【注入的状态信息】
{injection}

请从以下四个维度评分（1-5），只输出 JSON：

1. Recognition（识别）：文本中是否明确提到了注入的状态信息？
   - 5：明确提及，措辞清晰
   - 3：隐约暗示
   - 1：完全没有出现

2. Integration（整合）：状态信息是否被写入正文并成为场景的一部分？
   - 5：深度融入叙事，与场景自然交织
   - 3：被提及但孤立
   - 1：完全没有被整合

3. Utilization（利用）：状态信息是否影响了角色的决策或情节走向？
   - 5：成为决策的关键因素，影响剧情
   - 3：有轻微影响
   - 1：完全没有影响

4. Continuity（连续性）：场景之间的整体连续性如何？
   - 5：无缝衔接，逻辑连贯
   - 3：基本连贯但有轻微断裂
   - 1：明显断裂

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
            
            # 提取 JSON
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group())
                # 确保所有字段存在
                for key in ["recognition", "integration", "utilization", "continuity"]:
                    if key not in result:
                        result[key] = 3
                return result
            
            # 降级：提取数字
            nums = [int(x) for x in re.findall(r'[1-5]', content) if x.isdigit()]
            if len(nums) >= 4:
                return {
                    "recognition": nums[0],
                    "integration": nums[1],
                    "utilization": nums[2],
                    "continuity": nums[3],
                    "reason": "从文本提取数字"
                }
            return {
                "recognition": 3,
                "integration": 3,
                "utilization": 3,
                "continuity": 3,
                "reason": "解析失败"
            }
        except Exception as e:
            print(f"  [ERROR] {e}")
            return {
                "recognition": 3,
                "integration": 3,
                "utilization": 3,
                "continuity": 3,
                "reason": str(e)[:50]
            }


async def main():
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "reports" / "raw_m1"
    output_dir = base_dir / "reports" / "m1_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("机制评估 (固定版)")
    print("="*80)

    # 加载所有文本
    texts = load_texts(raw_dir)
    print(f"找到 {len(texts)} 个样本")

    # 按标签分组
    by_label = defaultdict(list)
    for t in texts:
        by_label[t["label"]].append(t)

    all_results = []

    for label in ["A1", "B1", "C1", "D1"]:
        items = by_label.get(label, [])
        if not items:
            print(f"警告: 没有找到 {label} 的样本")
            continue
        injection = INJECTION_DESC.get(label, "")
        print(f"\n--- {label} ({COND_MAP[label]}) ---")
        for item in items:
            print(f"  评估: {item['filename']}...")
            result = await call_judge(item["text"], injection)
            result["label"] = label
            result["rep"] = item["rep"]
            result["filename"] = item["filename"]
            all_results.append(result)
            print(f"    R:{result['recognition']} I:{result['integration']} U:{result['utilization']} C:{result['continuity']}")

    # 保存结果
    output_path = output_dir / "m1_scores_fixed.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")

    # 统计汇总
    print("\n" + "="*80)
    print("汇总统计")
    print("="*80)

    for label in ["A1", "B1", "C1", "D1"]:
        items = [r for r in all_results if r["label"] == label]
        if not items:
            continue
        n = len(items)
        r = statistics.mean([i["recognition"] for i in items])
        integ = statistics.mean([i["integration"] for i in items])
        u = statistics.mean([i["utilization"] for i in items])
        c = statistics.mean([i["continuity"] for i in items])
        print(f"\n{label} ({COND_MAP[label]}) n={n}:")
        print(f"  Recognition: {r:.2f}")
        print(f"  Integration: {integ:.2f}")
        print(f"  Utilization: {u:.2f}")
        print(f"  Continuity: {c:.2f}")

    # 关键对比
    print("\n" + "="*80)
    print("关键对比")
    print("="*80)
    a1 = [r for r in all_results if r["label"] == "A1"]
    d1 = [r for r in all_results if r["label"] == "D1"]
    if a1 and d1:
        a1_u = statistics.mean([i["utilization"] for i in a1])
        d1_u = statistics.mean([i["utilization"] for i in d1])
        a1_c = statistics.mean([i["continuity"] for i in a1])
        d1_c = statistics.mean([i["continuity"] for i in d1])
        print(f"A1 (GapHigh+MatchHigh) Utilization: {a1_u:.2f}")
        print(f"D1 (GapLow+MatchLow) Utilization: {d1_u:.2f}")
        print(f"Utilization 差异: {a1_u - d1_u:+.2f}")
        print(f"A1 Continuity: {a1_c:.2f}")
        print(f"D1 Continuity: {d1_c:.2f}")
        print(f"Continuity 差异: {a1_c - d1_c:+.2f}")


if __name__ == "__main__":
    asyncio.run(main())