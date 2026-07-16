#!/usr/bin/env python3
# experiments/phase5/pilot/evaluator.py
"""
LLM-as-Judge 评估器（最终修复版：支持 reasoning_content）
"""

import asyncio
import json
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import httpx
from dataclasses import dataclass

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"


@dataclass
class Sample:
    scene_id: str
    condition: str
    repeat: int
    text: str
    pair_id: str


@dataclass
class JudgeResult:
    scene_id: str
    condition: str
    repeat: int
    spatial_score: int
    brief_reason: str


def load_samples(raw_dir: Path) -> List[Sample]:
    """从 raw 目录加载所有样本（按位置解析文件名）"""
    samples = []
    for filepath in sorted(raw_dir.glob("*.txt")):
        parts = filepath.stem.split('_')
        if len(parts) < 7:
            print(f"  警告: 文件名格式异常: {filepath.name}")
            continue
        
        pair_id = f"{parts[0]}_{parts[1]}"  # pair_1
        scene_id = "_".join(parts[:4])      # pair_1_baseline_baseline
        condition = parts[3]                # baseline 或 intervention
        rep_str = parts[4]                  # rep00
        if rep_str.startswith("rep"):
            rep = int(rep_str[3:])
        else:
            print(f"  警告: 无法解析重复次数: {filepath.name}")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取正文
        text_match = re.search(r'={60}\n\n(.*?)\n\n={60}', content, re.DOTALL)
        if text_match:
            text = text_match.group(1).strip()
        else:
            text = content.strip()
        
        samples.append(Sample(
            scene_id=scene_id,
            condition=condition,
            repeat=rep,
            text=text,
            pair_id=pair_id
        ))
    
    return samples


def get_pair_a_text(pair_id: str) -> Optional[str]:
    """获取对应的 Scene A 文本"""
    base_dir = Path(__file__).parent
    scene_a_path = base_dir / "scenes" / pair_id / "scene_a.yaml"
    if scene_a_path.exists():
        with open(scene_a_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('text', '')
    return None


async def judge_sample(sample: Sample, scene_a_text: str, debug: bool = False) -> JudgeResult:
    """对单个样本进行 LLM 评估"""
    
    judge_prompt = f"""你是叙事连续性评估专家。

请阅读以下两段场景，判断【场景 B】是否自然承接【场景 A】的位置信息。

【场景 A】
{scene_a_text[:500]}

【场景 B】
{sample.text[:500]}

请根据 Spatial Continuity 标准评分（1-5）：
- 5：明确承接上一场景位置，过渡自然
- 4：位置一致，过渡稍显突兀
- 3：位置变化有交代，但缺乏细节
- 2：位置变化明显，仅有一句模糊交代
- 1：位置跳跃，没有任何解释

输出 JSON 格式：
{{
    "spatial_score": 1-5,
    "brief_reason": "简短理由"
}}"""

    async with httpx.AsyncClient(
        trust_env=False,
        timeout=httpx.Timeout(600.0, connect=30.0)
    ) as client:
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": judge_prompt}],
            "temperature": 0.1,
            "max_tokens": 512,  # 增大以确保有足够输出
        }
        
        try:
            response = await client.post(
                f"{LLM_API_BASE}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            
            # 优先取 content，若为空则取 reasoning_content
            content = data["choices"][0]["message"].get("content", "")
            if not content:
                content = data["choices"][0]["message"].get("reasoning_content", "")
            
            if debug:
                print(f"  [DEBUG] LLM 返回内容长度: {len(content)}")
                if content:
                    print(f"  [DEBUG] LLM 返回内容（前300字符）: {content[:300]}...")
                else:
                    print(f"  [DEBUG] LLM 返回内容为空!")
            
            if not content or len(content.strip()) < 10:
                print(f"  [WARNING] LLM 返回内容为空，使用默认值 3")
                return JudgeResult(
                    scene_id=sample.scene_id,
                    condition=sample.condition,
                    repeat=sample.repeat,
                    spatial_score=3,
                    brief_reason="LLM 返回为空"
                )
            
        except Exception as e:
            print(f"  [ERROR] API 调用失败: {e}")
            return JudgeResult(
                scene_id=sample.scene_id,
                condition=sample.condition,
                repeat=sample.repeat,
                spatial_score=3,
                brief_reason=f"API 错误: {str(e)[:50]}"
            )
        
        # 多种解析方式
        score = 3
        reason = "解析失败，使用默认值"
        
        # 方式1: 提取 JSON 对象
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if "spatial_score" in result:
                    score = int(result["spatial_score"])
                    reason = result.get("brief_reason", "")
                elif "score" in result:
                    score = int(result["score"])
                    reason = result.get("reason", "")
                elif "spatial" in result:
                    score = int(result["spatial"])
                    reason = result.get("reason", "")
                if debug:
                    print(f"  [DEBUG] JSON 解析成功: score={score}")
            except Exception as e:
                if debug:
                    print(f"  [DEBUG] JSON 解析失败: {e}")
        
        # 方式2: 提取带引号的 score
        if score == 3:
            match = re.search(r'"spatial_score"\s*[:：]\s*"?(\d)"?', content)
            if match:
                score = int(match.group(1))
        
        # 方式3: 提取 "评分: X"
        if score == 3:
            match = re.search(r'评分\s*[:：]\s*(\d)', content)
            if match:
                score = int(match.group(1))
        
        # 方式4: 提取 "score: X"
        if score == 3:
            match = re.search(r'score\s*[:：]\s*(\d)', content, re.IGNORECASE)
            if match:
                score = int(match.group(1))
        
        # 方式5: 提取任意 1-5 数字
        if score == 3:
            numbers = re.findall(r'\b([1-5])\b', content)
            if numbers:
                score = int(numbers[-1])
        
        # 钳制到 1-5
        score = max(1, min(5, score))
        
        return JudgeResult(
            scene_id=sample.scene_id,
            condition=sample.condition,
            repeat=sample.repeat,
            spatial_score=score,
            brief_reason=reason
        )


async def main():
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "reports" / "raw"
    output_dir = base_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("LLM-as-Judge 评估器（最终修复版）")
    print("="*60)
    
    debug = True
    samples = load_samples(raw_dir)
    print(f"找到 {len(samples)} 个样本")
    
    pairs = {}
    for s in samples:
        if s.pair_id not in pairs:
            pairs[s.pair_id] = []
        pairs[s.pair_id].append(s)
    
    print(f"找到 {len(pairs)} 个 pair")
    
    results = []
    for pair_id, pair_samples in pairs.items():
        print(f"\n评估 {pair_id}")
        scene_a_text = get_pair_a_text(pair_id)
        if not scene_a_text:
            print(f"  警告: 找不到 {pair_id}/scene_a.yaml")
            continue
        
        for sample in pair_samples:
            print(f"  评估 {sample.scene_id} (rep={sample.repeat})...")
            result = await judge_sample(sample, scene_a_text, debug=debug)
            results.append(result)
            print(f"    Score: {result.spatial_score} | {result.brief_reason}")
    
    # 保存结果
    results_data = [{
        "scene_id": r.scene_id,
        "condition": r.condition,
        "repeat": r.repeat,
        "spatial_score": r.spatial_score,
        "brief_reason": r.brief_reason
    } for r in results]
    
    output_path = output_dir / "llm_scores.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # 统计
    print("\n" + "="*60)
    print("统计结果")
    print("="*60)
    
    baseline_scores = [r.spatial_score for r in results if r.condition == "baseline"]
    intervention_scores = [r.spatial_score for r in results if r.condition == "intervention"]
    
    if baseline_scores:
        baseline_avg = sum(baseline_scores) / len(baseline_scores)
        print(f"Baseline 平均分: {baseline_avg:.2f} (n={len(baseline_scores)})")
    else:
        print("Baseline: 无数据")
    
    if intervention_scores:
        intervention_avg = sum(intervention_scores) / len(intervention_scores)
        print(f"Intervention 平均分: {intervention_avg:.2f} (n={len(intervention_scores)})")
    else:
        print("Intervention: 无数据")
    
    if baseline_scores and intervention_scores:
        delta = intervention_avg - baseline_avg
        print(f"\n差异: +{delta:.2f}")
        if delta > 0.5:
            print("✅ Location 干预对 Spatial Continuity 有显著提升")
        elif delta > 0.2:
            print("✅ Location 干预对 Spatial Continuity 有适度提升")
        else:
            print("⚠️ Location 干预对 Spatial Continuity 提升不明显")
    
    print(f"\n结果已保存到: {output_path}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())