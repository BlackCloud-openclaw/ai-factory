#!/usr/bin/env python3
# experiments/phase5/formal/validation_fidelity.py
"""
Evaluation Validation - Treatment Fidelity Check
检查 Writer 是否真正执行了 Physical State 干预
"""

import asyncio
import json
import re
from pathlib import Path
import httpx

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"


def load_texts(raw_dir: Path, pattern: str) -> list:
    """加载匹配 pattern 的文本文件"""
    texts = []
    for f in sorted(raw_dir.glob(pattern)):
        with open(f, 'r') as file:
            content = file.read()
        match = re.search(r'={60}\n\n(.*?)\n\n={60}', content, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            text = content[:500]
        texts.append({"file": f.name, "text": text})
    return texts


async def call_judge(text: str, debug: bool = False) -> dict:
    """调用 Judge，返回 Fidelity Score"""
    prompt = f"""请阅读以下场景，判断林逸的身体状态是否被持续描写为"受伤"或"身体异常"。

【场景 B】
{text}

请从以下维度评分（1-5）：
- 5：身体状态贯穿全文，影响行动和决策
- 4：身体状态在多处被提及，但未影响决策
- 3：身体状态在开头提及，之后消失
- 2：身体状态仅被暗示（如"喉头腥气"）
- 1：完全没有身体状态描写

只输出 JSON：
{{
    "fidelity_score": 1-5,
    "brief_reason": "简短理由"
}}"""

    async with httpx.AsyncClient(
        trust_env=False,
        timeout=httpx.Timeout(300.0, connect=30.0)
    ) as client:
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 200,
        }
        try:
            response = await client.post(
                f"{LLM_API_BASE}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"].get("content", "")
            if not content:
                content = data["choices"][0]["message"].get("reasoning_content", "")
            
            if debug:
                print(f"  [DEBUG] Judge 返回: {content[:100]}...")
            
            # 解析 JSON
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    score = result.get("fidelity_score", 3)
                    reason = result.get("brief_reason", "")
                    return {"score": score, "reason": reason, "raw": content}
                except:
                    pass
            
            # 尝试提取数字
            numbers = re.findall(r'\b([1-5])\b', content)
            if numbers:
                return {"score": int(numbers[0]), "reason": "从文本提取数字", "raw": content}
            
            return {"score": 3, "reason": "解析失败", "raw": content}
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            return {"score": 0, "reason": str(e), "raw": ""}


async def main():
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "reports" / "raw"
    
    print("="*80)
    print("Evaluation Validation - Treatment Fidelity Check")
    print("="*80)
    
    # 加载所有 C2 文本
    items = load_texts(raw_dir, "pair_*_C2_rep*.txt")
    print(f"找到 {len(items)} 个 C2 样本")
    
    results = []
    for item in items:
        print(f"\n评估: {item['file']}...")
        result = await call_judge(item['text'], debug=False)
        results.append({
            "file": item['file'],
            "fidelity_score": result['score'],
            "reason": result['reason'],
            "raw": result['raw']
        })
        print(f"  Fidelity Score: {result['score']}")
    
    # 统计
    print("\n" + "="*80)
    print("统计结果")
    print("="*80)
    
    scores = [r['fidelity_score'] for r in results if r['fidelity_score'] > 0]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"平均 Fidelity Score: {avg:.2f} (n={len(scores)})")
        print(f"分布:")
        for s in range(1, 6):
            count = sum(1 for x in scores if x == s)
            print(f"  {s}: {count}")
    else:
        print("无有效数据")
    
    # 判断
    if scores:
        high_fidelity = sum(1 for x in scores if x >= 4)
        low_fidelity = sum(1 for x in scores if x <= 2)
        
        print(f"\n高忠实度 (≥4): {high_fidelity}/{len(scores)}")
        print(f"低忠实度 (≤2): {low_fidelity}/{len(scores)}")
        
        if low_fidelity > high_fidelity:
            print("\n结论: Physical Intervention 未被 Writer 实际执行")
            print("→ 问题不在 Physical 无效，而在 Treatment Fidelity 不足")
        elif high_fidelity > low_fidelity:
            print("\n结论: Physical Intervention 被 Writer 较好执行")
            print("→ 如果 Blind Ranking 也显示差异，Physical 确实影响连续性")
        else:
            print("\n结论: Physical Intervention 被部分执行，但一致性不足")
    
    # 保存结果
    output_path = base_dir / "reports" / "validation_fidelity.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())