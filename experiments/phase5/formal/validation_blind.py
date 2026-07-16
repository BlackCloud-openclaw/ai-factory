#!/usr/bin/env python3
# experiments/phase5/formal/validation_blind.py
"""
Evaluation Validation - Blind Pairwise Ranking
测试 Judge 是否能在不知道实验条件的情况下区分 Baseline 和 C2
"""

import asyncio
import json
import re
from pathlib import Path
import httpx

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"

# pair_10 的 Scene A 文本
SCENE_A = """清晨的药园弥漫着露水与药草的气息，林逸提着竹篮穿梭在灵草间。他弯腰采下一株紫纹参时，指尖忽然触到异样的腥甜——本该清冽的药香里，混着一丝腐烂果核的腐臭。

他拨开半人高的玉灵草，发现三株血灵芝的叶片正蜷曲发黑，根部渗出暗褐色黏液。远处传来守药人的脚步声，他将血灵芝连根挖起，快步离开药园。"""


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
        texts.append(text)
    return texts


async def call_judge(text_a: str, text_b: str, debug: bool = False) -> dict:
    """调用 Judge，返回选择结果"""
    prompt = f"""请阅读以下三段场景，判断【场景 B1】和【场景 B2】哪一个更自然地承接【场景 A】的位置和状态。

【场景 A】
{SCENE_A}

【场景 B1】
{text_a}

【场景 B2】
{text_b}

请只输出一个选项：B1 或 B2。
不要解释，不要评分，只输出选项。"""

    async with httpx.AsyncClient(
        trust_env=False,
        timeout=httpx.Timeout(300.0, connect=30.0)
    ) as client:
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 20,
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
                print(f"  [DEBUG] Judge 返回: {content}")
            
            # 解析选择
            content_upper = content.strip().upper()
            if "B1" in content_upper and "B2" not in content_upper:
                return {"choice": "B1", "raw": content}
            elif "B2" in content_upper and "B1" not in content_upper:
                return {"choice": "B2", "raw": content}
            else:
                # 尝试提取数字
                numbers = re.findall(r'[12]', content)
                if numbers:
                    return {"choice": f"B{numbers[-1]}", "raw": content}
                return {"choice": "无法判断", "raw": content}
                
        except Exception as e:
            print(f"  [ERROR] {e}")
            return {"choice": "ERROR", "raw": str(e)}


async def main():
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "reports" / "raw"
    
    print("="*80)
    print("Evaluation Validation - Blind Pairwise Ranking")
    print("="*80)
    
    # 加载文本
    baseline_texts = load_texts(raw_dir, "pair_10_baseline_rep*.txt")
    c2_texts = load_texts(raw_dir, "pair_10_C2_rep*.txt")
    
    print(f"Baseline 文本数: {len(baseline_texts)}")
    print(f"C2 文本数: {len(c2_texts)}")
    
    results = []
    
    # 配对: 每个 Baseline 配一个 C2
    for i, (base, c2) in enumerate(zip(baseline_texts, c2_texts)):
        print(f"\n--- 配对 {i+1} ---")
        print(f"  Baseline 长度: {len(base)}")
        print(f"  C2 长度: {len(c2)}")
        
        # 随机打乱顺序 (但为了可复现，固定顺序)
        # 顺序: B1=Baseline, B2=C2
        print(f"  顺序: B1=Baseline, B2=C2")
        result = await call_judge(base, c2, debug=True)
        print(f"  Judge 选择: {result['choice']}")
        
        # 记录结果
        results.append({
            "pair": i + 1,
            "baseline_selected": result['choice'] == "B1",
            "c2_selected": result['choice'] == "B2",
            "choice": result['choice'],
            "raw": result['raw']
        })
    
    # 统计
    print("\n" + "="*80)
    print("统计结果")
    print("="*80)
    
    baseline_selected = sum(1 for r in results if r['baseline_selected'])
    c2_selected = sum(1 for r in results if r['c2_selected'])
    undecided = len(results) - baseline_selected - c2_selected
    
    print(f"Baseline 被选为更连续: {baseline_selected}/{len(results)}")
    print(f"C2 被选为更连续: {c2_selected}/{len(results)}")
    print(f"无法判断: {undecided}/{len(results)}")
    
    if baseline_selected > c2_selected:
        print("\n结论: Judge 偏向 Baseline → Physical 确实降低了连续性")
    elif c2_selected > baseline_selected:
        print("\n结论: Judge 偏向 C2 → Physical 可能提升了连续性")
    else:
        print("\n结论: Judge 无法区分 Baseline 和 C2 → 之前的 3.5 分可能是噪声")
    
    # 保存结果
    output_path = base_dir / "reports" / "validation_blind.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())