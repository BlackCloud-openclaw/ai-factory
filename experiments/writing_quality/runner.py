#!/usr/bin/env python
"""
Writing Quality POC - 离线实验工具（收敛版）

用法：
    python experiments/writing_quality/runner.py --chapter 16 --style xianhu
    python experiments/writing_quality/runner.py --all-chapters --style xianhu
"""

import asyncio
import argparse
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI
from experiments.writing_quality.metrics import MetricsObserver
from experiments.writing_quality.wqi_v1 import WQIV1
from experiments.writing_quality.validators import RewriteValidator

# 尝试导入生产 LLM 路由
try:
    from src.execution.llm_router_pool import get_llm_router_pool
    HAS_PRODUCTION_LLM = True
except ImportError:
    HAS_PRODUCTION_LLM = False


class WritingQualityExperiment:
    """写作质量离线实验运行器（收敛版）"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.prompts_dir = self.base_dir / "prompts"
        self.samples_dir = self.base_dir / "samples"
        self.reports_dir = self.base_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_PRODUCTION_LLM:
            self.pool = get_llm_router_pool()
        else:
            self.pool = None

    def load_prompt(self, style: str) -> Dict[str, str]:
        prompt_file = self.prompts_dir / f"{style}.yaml"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        with open(prompt_file, "r") as f:
            return yaml.safe_load(f)

    def load_chapter(self, chapter_num: int) -> str:
        sample_file = self.samples_dir / f"chap_{chapter_num:03d}.txt"
        if sample_file.exists():
            with open(sample_file, "r", encoding="utf-8") as f:
                return f.read()
        data_file = Path(f"data/novels/simple_long_novel_001/vol_001/chap_{chapter_num:03d}.txt")
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                return f.read()
        raise FileNotFoundError(f"Chapter {chapter_num} not found")

    async def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        if HAS_PRODUCTION_LLM:
            model = "Qwen3-32B-Q5_K_M-writer"
            async def _call(model_name: str, **kwargs) -> str:
                base_url = self.pool.get_base_url(model_name)
                client = AsyncOpenAI(api_key="not-needed", base_url=base_url)
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=8192,
                )
                return response.choices[0].message.content or ""
            try:
                return await self.pool.call(model, _call, timeout=300, agent="writing_experiment")
            except Exception as e:
                print(f"LLM call failed: {e}")
                return ""
        else:
            import os
            api_url = os.environ.get("LLM_API_URL", "http://localhost:8082")
            client = AsyncOpenAI(api_key="not-needed", base_url=api_url)
            response = await client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "Qwen3-32B-Q5_K_M-writer"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=8192,
            )
            return response.choices[0].message.content or ""

    def extract_must_events(self, chapter_num: int) -> List[str]:
        """从大纲中提取 must_events（简化版，从章节号推断）"""
        # 第5章：通过入门石碑考验，分配外门住所
        # 第16章：发现隐秘灵泉，遭遇多方围堵
        # 第18章：剑阁试炼，触发剑阵
        # 第20章：感应天地异象，发现筑基隐患
        events_map = {
            5: ["入门石碑考验", "分配外门住所"],
            16: ["发现隐秘灵泉", "遭遇多方围堵"],
            18: ["触发剑阵", "发现剑阁禁制"],
            20: ["感应天地异象", "发现筑基隐患"],
        }
        return events_map.get(chapter_num, [])

    async def run_experiment(self, chapter_num: int, style: str) -> Dict[str, Any]:
        print(f"\n📖 第 {chapter_num} 章 | 风格: {style}")

        # 1. 加载原文
        original = self.load_chapter(chapter_num)
        print(f"  原文长度: {len(original)} 字符")

        # 2. 加载 Prompt
        prompt_config = self.load_prompt(style)
        system_prompt = prompt_config.get("system_prompt", "")
        user_template = prompt_config.get("user_template", "")
        user_prompt = user_template.format(original_text=original)

        # 3. 调用 LLM
        print("  正在调用 LLM...")
        rewritten = await self.call_llm(system_prompt, user_prompt)
        if not rewritten:
            print("  ❌ LLM 返回空内容")
            return None
        print(f"  改写长度: {len(rewritten)} 字符")

        # 4. 验证
        must_events = self.extract_must_events(chapter_num)
        validation = RewriteValidator.validate(original, rewritten, must_events)
        print(f"  验证结果: {'✅ 通过' if validation['passed'] else '❌ 失败'}")
        if validation['issues']:
            for issue in validation['issues']:
                print(f"    - {issue}")

        # 5. WQI V1 评分
        orig_wqi = WQIV1.score(original, original)
        new_wqi = WQIV1.score(rewritten, original)
        wqi_gain = new_wqi["total"] - orig_wqi["total"]
        print(f"  WQI: {orig_wqi['total']:.1f} → {new_wqi['total']:.1f} (+{wqi_gain:.1f})")

        # 6. 指标对比
        metrics_compare = MetricsObserver.compare(original, rewritten)

        # 7. 保存结果
        self.save_result(chapter_num, style, original, rewritten, validation, orig_wqi, new_wqi, metrics_compare)

        print(f"  ✅ 结果保存至: {self.reports_dir}")

        return {
            "chapter": chapter_num,
            "style": style,
            "original": original,
            "rewritten": rewritten,
            "validation": validation,
            "orig_wqi": orig_wqi,
            "new_wqi": new_wqi,
            "wqi_gain": wqi_gain,
            "metrics_compare": metrics_compare,
        }

    def save_result(self, chapter_num: int, style: str, original: str, rewritten: str,
                    validation: Dict, orig_wqi: Dict, new_wqi: Dict, metrics_compare: Dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.reports_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / f"chap_{chapter_num:03d}_{style}_result.json", "w", encoding="utf-8") as f:
            json.dump({
                "chapter": chapter_num,
                "style": style,
                "timestamp": timestamp,
                "validation": validation,
                "orig_wqi": orig_wqi,
                "new_wqi": new_wqi,
                "wqi_gain": new_wqi["total"] - orig_wqi["total"],
                "metrics_compare": metrics_compare,
            }, f, ensure_ascii=False, indent=2)

        with open(run_dir / f"chap_{chapter_num:03d}_{style}_original.txt", "w", encoding="utf-8") as f:
            f.write(original)
        with open(run_dir / f"chap_{chapter_num:03d}_{style}_rewritten.txt", "w", encoding="utf-8") as f:
            f.write(rewritten)

        # 生成报告
        report_lines = [
            f"# 写作质量实验报告",
            f"\n**章节**: {chapter_num}",
            f"**风格**: {style}",
            f"**时间**: {timestamp}",
            f"\n## 验证结果",
            f"- 通过: {'✅' if validation['passed'] else '❌'}",
            f"- 事件保留率: {validation['event_retention']*100:.1f}%",
            f"- 角色一致性: {validation['character_consistency']*100:.1f}%",
            f"- 字数倍率: {validation['length_ratio']:.2f}",
            f"\n## WQI V1",
            f"- 原文: {orig_wqi['total']:.1f}",
            f"- 改写: {new_wqi['total']:.1f}",
            f"- 提升: {new_wqi['total'] - orig_wqi['total']:+.1f}",
            f"\n### 各维度得分",
        ]
        for k, v in new_wqi["scores"].items():
            report_lines.append(f"- {k}: {v:.1f}")
        report_lines.append("\n## 指标对比")
        report_lines.append(f"- 对话占比: {metrics_compare['deltas']['dialogue_ratio']*100:+.1f}%")
        report_lines.append(f"- 内心活动: {metrics_compare['deltas']['inner_monologue_density']*100:+.1f}%")
        report_lines.append(f"- 感官密度: {metrics_compare['deltas']['sensory_density']*100:+.1f}%")
        report_lines.append(f"- 字数变化: {metrics_compare['deltas']['total_chars']:+d}")

        with open(run_dir / f"chap_{chapter_num:03d}_{style}_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

    async def run_all_chapters(self, style: str, chapters: List[int]):
        """运行指定风格的所有章节"""
        results = []
        for chapter in chapters:
            result = await self.run_experiment(chapter, style)
            if result:
                results.append(result)
        self.generate_summary(results, style)

    def generate_summary(self, results: List[Dict], style: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.reports_dir / f"summary_{style}_{timestamp}.md"

        lines = [
            f"# 写作质量实验汇总",
            f"\n**风格**: {style}",
            f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## 结果概览\n",
            "| 章节 | WQI 提升 | 事件保留率 | 角色一致性 | 字数倍率 | 通过 |",
            "|------|---------|-----------|-----------|---------|------|",
        ]

        for r in results:
            v = r["validation"]
            gain = r["wqi_gain"]
            passed = "✅" if v["passed"] else "❌"
            lines.append(
                f"| {r['chapter']} | +{gain:.1f} | {v['event_retention']*100:.0f}% | "
                f"{v['character_consistency']*100:.0f}% | {v['length_ratio']:.2f}x | {passed} |"
            )

        lines.append("\n## 判断")
        all_passed = all(r["validation"]["passed"] for r in results)
        avg_gain = sum(r["wqi_gain"] for r in results) / len(results) if results else 0

        if all_passed and avg_gain >= 20:
            lines.append("\n✅ **实验成功**: 所有样本通过验证，WQI 平均提升 ≥ 20 分，可进入下一轮。")
        else:
            lines.append(f"\n⚠️ **需要调整**: 通过率 {sum(1 for r in results if r['validation']['passed'])/len(results)*100:.0f}%，平均 WQI 提升 {avg_gain:.1f} 分。")

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\n📊 汇总报告: {summary_file}")


async def main():
    parser = argparse.ArgumentParser(description="Writing Quality POC (收敛版)")
    parser.add_argument("--chapter", type=int, help="测试章节号（05, 16, 18, 20）")
    parser.add_argument("--style", default="xianhu", choices=["xianhu", "dialogue_heavy", "psychological", "immersive"], help="风格")
    parser.add_argument("--all-chapters", action="store_true", help="测试所有章节（05,16,18,20）")

    args = parser.parse_args()

    if not any([args.chapter, args.all_chapters]):
        parser.print_help()
        return

    chapters = [5, 16, 18, 20] if args.all_chapters else [args.chapter]

    experiment = WritingQualityExperiment()
    await experiment.run_all_chapters(args.style, chapters)


if __name__ == "__main__":
    asyncio.run(main())