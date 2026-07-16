#!/usr/bin/env python3
"""
TR 测量脚本 - 使用概率分配格式（稳定版本）
"""

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Dict, List
from collections import Counter
from datetime import datetime
import httpx

# ============================================================
# 配置
# ============================================================

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"
SAMPLES = 10
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120.0
REQUEST_DELAY = 2.0
CONNECT_TIMEOUT = 10.0

OUTPUT_DIR = Path(__file__).parent / "reports" / "tr_measurement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4 个测试场景
# ============================================================

SCENES = [
    {
        "id": "scene_01",
        "name": "关系型 - 十年师兄重逢",
        "scene_a": """林逸抬起头，看见一个熟悉的身影站在不远处。那人背对着月光，轮廓模糊，但他绝不会认错——那是十年前消失的师兄。两人之间隔着十几步，谁也没有先开口。夜风从两人之间穿过，吹动衣角。""",
        "options": [
            "快步上前，质问师兄当年为何失踪",
            "站在原地，等师兄先开口",
            "转身离开，暂不面对这段往事",
            "试探性地喊出师兄的名字"
        ]
    },
    {
        "id": "scene_02",
        "name": "伦理型 - 两难选择",
        "scene_a": """林逸站在岔路口，左边的路通向宗门，右边的路通向禁地。师父交代的任务是三天内返回宗门复命，但禁地方向隐约传来异样的灵力波动。他停下脚步，望着两条路。""",
        "options": [
            "立即返回宗门，遵守师命",
            "立即前往禁地，探查灵力波动",
            "立即在原地留下标记，等待同伴",
            "立即登上高处，观察禁地方向再决定"
        ]
    },
    {
        "id": "scene_03",
        "name": "社交型 - 议事堂争端",
        "scene_a": """林逸转身踏出议事堂门槛，晚风裹着青草气扑面而来。身后传来管事的茶杯碎裂声，他没有回头，沿着石阶朝外走去。脚步在青石板上发出清脆的回响，石阶尽头分出三条路。""",
        "options": [
            "回到自己住处，独处思考",
            "去师弟住处，寻求商议",
            "直接去找长老，陈述此事",
            "离开宗门，暂避风头"
        ]
    },
    {
        "id": "scene_04",
        "name": "信息型 - 密信",
        "scene_a": """林逸从书册夹层中抽出一封泛黄的密信。封蜡已经碎裂，但他认出了那个印记——那是宗门长老的私印。信纸只有半张，字迹潦草。他读完后，指尖微微发抖，将信纸轻轻放在桌面上。门外的脚步声正在靠近。""",
        "options": [
            "将密信藏入怀中，寻找更多线索",
            "将密信留在桌面上，装作没发现",
            "立即去找可信的长辈商议",
            "将密信烧掉，毁掉证据"
        ]
    }
]


# ============================================================
# TR 测量类（概率分配版本）
# ============================================================

class TRMeasurement:
    def __init__(self):
        self.results = {}
    
    async def _call_llm(self, prompt: str, retries: int = MAX_RETRIES) -> str:
        """调用 LLM API，带重试"""
        last_error = None
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(
                    trust_env=False,
                    timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)
                ) as client:
                    payload = {
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 512,
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
                    last_error = "Empty response"
            except Exception as e:
                last_error = str(e)
            if attempt < retries - 1:
                await asyncio.sleep((attempt + 1) * 2)
        print(f"  [FATAL] All retries failed: {last_error}")
        return ""
    
    def _extract_probs(self, response: str) -> Dict[str, float]:
        """从响应中提取四个选项的概率"""
        if not response or not response.strip():
            return {}
        
        probs = {}
        for opt_id in ["A", "B", "C", "D"]:
            patterns = [
                rf'{opt_id}[:：]\s*(\d+)%',
                rf'{opt_id}\s*[:：]\s*(\d+\.?\d*)%',
                rf'{opt_id}\s*[:：]\s*(\d+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    probs[opt_id] = float(match.group(1)) / 100.0
                    break
            if opt_id not in probs:
                probs[opt_id] = 0.25
        
        # 归一化
        total = sum(probs.values())
        if total > 0:
            for k in probs:
                probs[k] = probs[k] / total
        else:
            probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        
        return probs
    
    async def measure_scene(self, scene: Dict, samples: int = SAMPLES) -> Dict:
        """测量单个场景的 TR"""
        scene_id = scene["id"]
        scene_name = scene["name"]
        scene_a = scene["scene_a"]
        options = scene["options"]
        
        print(f"\n测量: {scene_name} ({scene_id})")
        print(f"  样本数: {samples}")
        
        prompt_template = f"""你是一位小说情节分析师。请阅读以下场景结尾。

【场景结尾】
{scene_a}

【任务】
判断以下四个后续场景的合理性分配。
四个方案都同样合理，没有标准答案。

A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

请为每个选项分配概率（0-100%），总和=100%。

输出格式：
A: XX%
B: XX%
C: XX%
D: XX%"""

        all_probs = []
        raw_responses = []
        
        for s in range(samples):
            print(f"  样本 {s+1}/{samples}...")
            response = await self._call_llm(prompt_template)
            raw_responses.append(response)
            probs = self._extract_probs(response)
            if probs and sum(probs.values()) > 0:
                all_probs.append(probs)
                print(f"    概率: A={probs['A']:.0%}, B={probs['B']:.0%}, C={probs['C']:.0%}, D={probs['D']:.0%}")
            else:
                print(f"    警告: 无法解析概率")
            await asyncio.sleep(REQUEST_DELAY)
        
        n_valid = len(all_probs)
        print(f"  有效样本: {n_valid}/{samples}")
        
        if n_valid == 0:
            return {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "valid_samples": 0,
                "error": "No valid probabilities",
                "raw_responses": raw_responses
            }
        
        # Top-1 分布（基于概率最高的选项）
        top1s = [max(p, key=p.get) for p in all_probs]
        top1_counter = Counter(top1s)
        tr = top1_counter.most_common(1)[0][1] / n_valid if top1_counter else 0
        
        # 平均概率
        avg_probs = {}
        for opt in ["A", "B", "C", "D"]:
            avg_probs[opt] = sum(p[opt] for p in all_probs) / n_valid
        
        # 熵（基于 Top-1 分布）
        probs_dist = [count / n_valid for count in top1_counter.values()]
        entropy = -sum(p * math.log2(p) for p in probs_dist) if probs_dist else 0
        
        # 胜者边际
        if len(top1_counter) >= 2:
            sorted_counts = sorted(top1_counter.values(), reverse=True)
            winner_margin = (sorted_counts[0] - sorted_counts[1]) / n_valid
        else:
            winner_margin = 1.0
        
        # 不同 Top-1 数量
        distinct_top1 = len(top1_counter)
        
        return {
            "scene_id": scene_id,
            "scene_name": scene_name,
            "valid_samples": n_valid,
            "samples_requested": samples,
            "TR": tr,
            "entropy": entropy,
            "winner_margin": winner_margin,
            "distinct_top1": distinct_top1,
            "distribution": dict(top1_counter),
            "avg_probabilities": avg_probs,
            "top1_sequence": top1s,
            "all_probabilities": all_probs,
            "raw_responses": raw_responses
        }
    
    def generate_report(self, results: List[Dict]) -> str:
        """生成 Markdown 报告"""
        lines = []
        lines.append("# TR 测量报告（概率分配版本）")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().isoformat()}")
        lines.append(f"**模型**: {LLM_MODEL}")
        lines.append(f"**每个场景样本数**: {SAMPLES}")
        lines.append("")
        
        # 汇总表
        lines.append("## 汇总")
        lines.append("")
        lines.append("| 场景 | TR | 熵 | 胜者边际 | 不同 Top-1 | 分布 |")
        lines.append("|------|-----|-----|----------|------------|------|")
        
        for r in results:
            if "error" in r:
                continue
            dist_str = " / ".join([f"{k}:{v}" for k, v in r["distribution"].items()])
            lines.append(
                f"| {r['scene_name'][:10]} | {r['TR']:.2f} | {r['entropy']:.2f} | "
                f"{r['winner_margin']:.2f} | {r['distinct_top1']} | {dist_str} |"
            )
        lines.append("")
        
        # 平均概率表
        lines.append("## 平均概率")
        lines.append("")
        lines.append("| 场景 | A | B | C | D |")
        lines.append("|------|---|---|---|---|")
        for r in results:
            if "error" in r:
                continue
            avg = r["avg_probabilities"]
            lines.append(
                f"| {r['scene_name'][:10]} | {avg['A']:.0%} | {avg['B']:.0%} | "
                f"{avg['C']:.0%} | {avg['D']:.0%} |"
            )
        lines.append("")
        
        # 分类
        lines.append("## 分类")
        lines.append("")
        lines.append("| 分类 | 标准 | 场景 |")
        lines.append("|------|------|------|")
        
        rigid = []
        moderate = []
        competitive = []
        open_scenes = []
        
        for r in results:
            if "error" in r:
                continue
            name = r['scene_name'][:15]
            tr = r['TR']
            if tr > 0.85:
                rigid.append(name)
            elif 0.60 <= tr <= 0.85:
                moderate.append(name)
            elif 0.35 <= tr < 0.60:
                competitive.append(name)
            else:
                open_scenes.append(name)
        
        if rigid:
            lines.append(f"| **Rigid** (TR > 0.85) | 不适合 State 操纵 | {', '.join(rigid)} |")
        if moderate:
            lines.append(f"| **Moderately Rigid** (0.60-0.85) | 边界条件 | {', '.join(moderate)} |")
        if competitive:
            lines.append(f"| **Competitive** (0.35-0.60) | 优先用于 State 实验 | {', '.join(competitive)} |")
        if open_scenes:
            lines.append(f"| **Highly Open** (TR < 0.35) | 适合 State 主导预测 | {', '.join(open_scenes)} |")
        lines.append("")
        
        # 诊断
        lines.append("## 诊断")
        lines.append("")
        if not rigid and not moderate and not competitive and not open_scenes:
            lines.append("**无有效数据。**")
            lines.append("→ 检查 LLM 服务是否正常。")
        elif not rigid and not moderate:
            lines.append("**所有场景均为 Competitive 或 Highly Open。**")
            lines.append("→ State 操纵有较大空间，可以继续 Phase 5.2。")
        elif len(competitive) >= 2:
            lines.append(f"**有 {len(competitive)} 个 Competitive 场景。**")
            lines.append("→ 可以优先使用这些场景进行 Phase 5.2。")
        elif len(moderate) >= 3:
            lines.append(f"**有 {len(moderate)} 个 Moderately Rigid 场景。**")
            lines.append("→ 建议先修改这些场景的设计，增加价值冲突。")
        else:
            lines.append("**所有场景均为 Rigid。**")
            lines.append("→ 问题可能出在测量任务本身（概率分配），而非场景设计。")
        
        return "\n".join(lines)
    
    async def run_all(self):
        """运行所有场景的测量"""
        print("=" * 60)
        print("TR 测量（概率分配版本）: 4 个场景 × 10 次")
        print("=" * 60)
        print(f"模型: {LLM_MODEL}")
        print(f"每个场景: {SAMPLES} 次")
        print("")
        
        results = []
        for scene in SCENES:
            result = await self.measure_scene(scene, samples=SAMPLES)
            results.append(result)
            print(f"  完成: {result['scene_name']} - TR = {result.get('TR', 'N/A')}")
        
        # 保存原始数据
        raw_path = OUTPUT_DIR / "raw_data.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n原始数据保存到: {raw_path}")
        
        # 生成报告
        report = self.generate_report(results)
        report_path = OUTPUT_DIR / "report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"报告保存到: {report_path}")
        
        print("\n" + "=" * 60)
        print("完成")
        print("=" * 60)


async def main():
    measurement = TRMeasurement()
    await measurement.run_all()

if __name__ == "__main__":
    asyncio.run(main())