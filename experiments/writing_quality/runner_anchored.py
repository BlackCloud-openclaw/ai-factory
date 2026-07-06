#!/usr/bin/env python
"""
锚定重写 Runner（字数强制版）
"""

import asyncio
import argparse
import sys
import yaml
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI

from experiments.writing_quality.metrics import MetricsObserver
from experiments.writing_quality.wqi_v1 import WQIV1
from experiments.writing_quality.anchor_extractor import AnchorExtractor
from experiments.writing_quality.anchor_validator import AnchorValidator

try:
    from src.execution.llm_router_pool import get_llm_router_pool
    HAS_PRODUCTION_LLM = True
except ImportError:
    HAS_PRODUCTION_LLM = False


class AnchoredRewriteExperiment:
    """锚定重写实验运行器（字数强制版）"""

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

    def load_prompt(self) -> Dict[str, str]:
        prompt_file = self.prompts_dir / "xianhu_anchored.yaml"
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
                    max_tokens=16384,
                )
                return response.choices[0].message.content or ""
            try:
                return await self.pool.call(model, _call, timeout=600, agent="anchored_rewrite")
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
                max_tokens=16384,
            )
            return response.choices[0].message.content or ""

    def build_prompt(self, original: str, anchors: Dict[str, List[str]]) -> tuple:
        """构建锚定 Prompt（含字数强制）"""
        prompt_config = self.load_prompt()
        system_template = prompt_config.get("system_prompt", "")
        user_template = prompt_config.get("user_template", "")

        original_length = len(original)
        target_length = int(original_length * 2.5)

        character_list = "、".join(anchors.get("characters", [])) or "（无）"
        item_list = "、".join(anchors.get("items", [])) or "（无）"
        realm_list = "、".join(anchors.get("realms", [])) or "（无）"

        system_prompt = system_template.format(
            character_list=character_list,
            item_list=item_list,
            realm_list=realm_list,
            original_length=original_length,
            target_length=target_length,
        )

        user_prompt = user_template.format(
            original_text=original,
            character_list=character_list,
            item_list=item_list,
            realm_list=realm_list,
            original_length=original_length,
            target_length=target_length,
        )

        return system_prompt, user_prompt

    def extract_anchors(self, text: str) -> Dict[str, List[str]]:
        return AnchorExtractor.extract_all(text)

    def validate_result(self, original: str, rewritten: str, anchors: Dict[str, List[str]]) -> Dict[str, Any]:
        return AnchorValidator.validate(original, rewritten, anchors)

    async def run_experiment(self, chapter_num: int) -> Dict[str, Any]:
        print(f"\n📖 第 {chapter_num} 章")

        original = self.load_chapter(chapter_num)
        original_len = len(original)
        target_len = int(original_len * 2.5)
        print(f"  原文长度: {original_len} 字符")
        print(f"  目标长度: {target_len} 字符 (2.5x)")

        anchors = self.extract_anchors(original)
        print(f"  锚点: {len(anchors.get('characters', []))} 个角色, "
              f"{len(anchors.get('items', []))} 个物品, "
              f"{len(anchors.get('realms', []))} 个境界")

        system_prompt, user_prompt = self.build_prompt(original, anchors)

        print("  正在调用 LLM...")
        rewritten = await self.call_llm(system_prompt, user_prompt)
        if not rewritten:
            print("  ❌ LLM 返回空内容")
            return None

        rewritten_len = len(rewritten)
        length_ratio = rewritten_len / original_len
        print(f"  改写长度: {rewritten_len} 字符 ({length_ratio:.2f}x)")

        # 字数达标检查
        if rewritten_len < target_len * 0.8:
            print(f"  ⚠️ 字数不足目标 ({rewritten_len}/{target_len})，仍保存但标记")
            length_ok = False
        else:
            length_ok = True

        validation = self.validate_result(original, rewritten, anchors)
        print(f"  验证结果: {'✅ 通过' if validation['passed'] else '❌ 失败'}")
        if validation['issues']:
            for issue in validation['issues']:
                print(f"    - {issue}")

        orig_wqi = WQIV1.score(original, original)
        new_wqi = WQIV1.score(rewritten, original)
        wqi_gain = new_wqi["total"] - orig_wqi["total"]
        print(f"  WQI: {orig_wqi['total']:.1f} → {new_wqi['total']:.1f} ({wqi_gain:+.1f})")

        metrics_compare = MetricsObserver.compare(original, rewritten)

        self.save_result(chapter_num, original, rewritten, anchors, validation, orig_wqi, new_wqi, metrics_compare, length_ok)

        print(f"  ✅ 结果保存至: {self.reports_dir}")

        return {
            "chapter": chapter_num,
            "original": original,
            "rewritten": rewritten,
            "original_length": original_len,
            "rewritten_length": rewritten_len,
            "length_ratio": length_ratio,
            "length_ok": length_ok,
            "anchors": anchors,
            "validation": validation,
            "orig_wqi": orig_wqi,
            "new_wqi": new_wqi,
            "wqi_gain": wqi_gain,
            "metrics_compare": metrics_compare,
        }

    def save_result(self, chapter_num: int, original: str, rewritten: str,
                    anchors: Dict, validation: Dict, orig_wqi: Dict, new_wqi: Dict,
                    metrics_compare: Dict, length_ok: bool):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.reports_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / f"chap_{chapter_num:03d}_result.json", "w", encoding="utf-8") as f:
            json.dump({
                "chapter": chapter_num,
                "timestamp": timestamp,
                "original_length": len(original),
                "rewritten_length": len(rewritten),
                "length_ratio": len(rewritten) / len(original),
                "length_ok": length_ok,
                "anchors": anchors,
                "validation": validation,
                "orig_wqi": orig_wqi,
                "new_wqi": new_wqi,
                "wqi_gain": new_wqi["total"] - orig_wqi["total"],
                "metrics_compare": metrics_compare,
            }, f, ensure_ascii=False, indent=2)

        with open(run_dir / f"chap_{chapter_num:03d}_original.txt", "w", encoding="utf-8") as f:
            f.write(original)
        with open(run_dir / f"chap_{chapter_num:03d}_rewritten.txt", "w", encoding="utf-8") as f:
            f.write(rewritten)

        report_lines = [
            f"# 锚定重写实验报告（字数强制）",
            f"\n**章节**: {chapter_num}",
            f"**时间**: {timestamp}",
            f"\n## 字数",
            f"- 原文: {len(original)} 字符",
            f"- 改写: {len(rewritten)} 字符",
            f"- 倍率: {len(rewritten)/len(original):.2f}x",
            f"- 目标达标: {'✅' if length_ok else '❌'}",
            f"\n## 锚点",
            f"- 角色: {', '.join(anchors.get('characters', []))}",
            f"- 物品: {', '.join(anchors.get('items', []))}",
            f"- 境界: {', '.join(anchors.get('realms', []))}",
            f"\n## 验证结果",
            f"- 通过: {'✅' if validation['passed'] else '❌'}",
        ]
        for key, detail in validation.get('details', {}).items():
            report_lines.append(f"- {key}: {detail['found']}/{detail['total']}")
        report_lines.append(f"\n## WQI")
        report_lines.append(f"- 原文: {orig_wqi['total']:.1f}")
        report_lines.append(f"- 改写: {new_wqi['total']:.1f}")
        report_lines.append(f"- 提升: {new_wqi['total'] - orig_wqi['total']:+.1f}")

        with open(run_dir / f"chap_{chapter_num:03d}_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

    async def run_all_chapters(self, chapters: List[int]):
        results = []
        for chapter in chapters:
            result = await self.run_experiment(chapter)
            if result:
                results.append(result)
        self.generate_summary(results)

    def generate_summary(self, results: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.reports_dir / f"summary_anchored_{timestamp}.md"

        lines = [
            f"# 锚定重写实验汇总（字数强制）",
            f"\n**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## 结果概览\n",
            "| 章节 | 原文 | 改写 | 倍率 | WQI提升 | 字数达标 | 锚点通过 |",
            "|------|------|------|------|---------|---------|----------|",
        ]

        for r in results:
            lines.append(
                f"| {r['chapter']} | {r['original_length']} | {r['rewritten_length']} | "
                f"{r['length_ratio']:.2f}x | {r['wqi_gain']:+.1f} | "
                f"{'✅' if r['length_ok'] else '❌'} | {'✅' if r['validation']['passed'] else '❌'} |"
            )

        all_length_ok = all(r.get('length_ok', False) for r in results)
        all_anchor_ok = all(r['validation']['passed'] for r in results)
        avg_gain = sum(r["wqi_gain"] for r in results) / len(results) if results else 0

        lines.append(f"\n## 判断")
        if all_length_ok and all_anchor_ok and avg_gain >= 15:
            lines.append("\n✅ **实验成功**: 字数、锚点、WQI 全部达标。")
        else:
            lines.append(f"\n⚠️ **需要调整**: 字数达标率 {sum(1 for r in results if r.get('length_ok', False))/len(results)*100:.0f}%，")
            lines.append(f"锚点通过率 {sum(1 for r in results if r['validation']['passed'])/len(results)*100:.0f}%，")
            lines.append(f"平均 WQI 提升 {avg_gain:.1f} 分。")

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\n📊 汇总报告: {summary_file}")


async def main():
    parser = argparse.ArgumentParser(description="锚定重写实验（字数强制版）")
    parser.add_argument("--chapter", type=int, help="测试章节号")
    parser.add_argument("--all-chapters", action="store_true", help="测试所有章节")

    args = parser.parse_args()

    if not any([args.chapter, args.all_chapters]):
        parser.print_help()
        return

    chapters = [16, 18, 20] if args.all_chapters else [args.chapter]

    experiment = AnchoredRewriteExperiment()
    await experiment.run_all_chapters(chapters)


if __name__ == "__main__":
    asyncio.run(main())