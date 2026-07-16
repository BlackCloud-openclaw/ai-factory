#!/usr/bin/env python3
"""
Phase 5.2: Writer 实验
使用关系型重逢场景（TR=0.33），测试 State 对生成实现的影响
"""

import asyncio
import json
import re
from pathlib import Path
from datetime import datetime
import httpx

# ============================================================
# 配置
# ============================================================

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"
SAMPLES_PER_CONDITION = 5
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120.0
REQUEST_DELAY = 2.0

OUTPUT_DIR = Path(__file__).parent / "reports" / "phase5_2_writer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 场景材料
# ============================================================

SCENE_A = """林逸抬起头，看见一个熟悉的身影站在不远处。那人背对着月光，轮廓模糊，但他绝不会认错——那是十年前消失的师兄。两人之间隔着十几步，谁也没有先开口。夜风从两人之间穿过，吹动衣角。"""

CONDITIONS = {
    "control": {
        "name": "Control（无 State）",
        "state": "",
        "description": "请从以上场景继续，写一段林逸与师兄重逢的后续场景。"
    },
    "state_trust": {
        "name": "State A（高信任）",
        "state": "林逸想起师兄失踪前曾留给他一封密信，信中暗示自己要去调查宗门内部的一个秘密。",
        "description": "请从以上场景继续，写一段林逸与师兄重逢的后续场景。林逸记得师兄失踪前留下的密信内容。"
    },
    "state_conflict": {
        "name": "State B（高冲突）",
        "state": "林逸知道宗门密档中有一份指控师兄叛逃的记录。",
        "description": "请从以上场景继续，写一段林逸与师兄重逢的后续场景。林逸知道宗门密档中对师兄的指控。"
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
2. 描写人物之间的互动、对话和内心活动
3. 自然过渡到后续情节
4. 不要添加任何解释或元评论
5. 只输出续写的正文，不要包含任何额外内容"""

# ============================================================
# Writer 实验
# ============================================================

class Phase52Writer:
    def __init__(self):
        self.results = []
    
    async def _call_llm(self, prompt: str, retries: int = MAX_RETRIES) -> str:
        """调用 LLM API，带重试"""
        last_error = None
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(
                    trust_env=False,
                    timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0)
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
                    last_error = "Empty response"
            except Exception as e:
                last_error = str(e)
            if attempt < retries - 1:
                await asyncio.sleep((attempt + 1) * 2)
        print(f"  [FATAL] All retries failed: {last_error}")
        return ""
    
    async def run_condition(self, condition_key: str, samples: int = SAMPLES_PER_CONDITION):
        """运行单个条件"""
        cond = CONDITIONS[condition_key]
        print(f"\n运行: {cond['name']}")
        print(f"  样本数: {samples}")
        
        results = []
        for s in range(samples):
            print(f"  样本 {s+1}/{samples}...")
            
            # 构建 Prompt
            state_text = f"\n【状态信息】\n{cond['state']}" if cond['state'] else ""
            prompt = WRITER_PROMPT_TEMPLATE.format(
                scene_a=SCENE_A,
                description=cond['description'],
                state=state_text
            )
            
            # 调用 LLM
            response = await self._call_llm(prompt)
            
            # 记录
            results.append({
                "sample": s,
                "condition": condition_key,
                "condition_name": cond['name'],
                "state": cond['state'],
                "text": response,
                "length": len(response)
            })
            print(f"    长度: {len(response)} 字符")
            await asyncio.sleep(REQUEST_DELAY)
        
        return results
    
    async def run_all(self):
        """运行所有条件"""
        print("=" * 60)
        print("Phase 5.2 Writer 实验")
        print("=" * 60)
        print(f"模型: {LLM_MODEL}")
        print(f"每个条件样本数: {SAMPLES_PER_CONDITION}")
        print("")
        
        all_results = []
        
        for cond_key in ["control", "state_trust", "state_conflict"]:
            results = await self.run_condition(cond_key)
            all_results.extend(results)
        
        # 保存数据
        raw_path = OUTPUT_DIR / "raw_data.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n原始数据保存到: {raw_path}")
        
        # 生成报告
        report = self.generate_report(all_results)
        report_path = OUTPUT_DIR / "report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"报告保存到: {report_path}")
        
        print("\n" + "=" * 60)
        print("完成")
        print("=" * 60)
        
        return all_results
    
    def generate_report(self, results: list) -> str:
        """生成报告"""
        lines = []
        lines.append("# Phase 5.2 Writer 实验报告")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().isoformat()}")
        lines.append(f"**模型**: {LLM_MODEL}")
        lines.append(f"**每个条件样本数**: {SAMPLES_PER_CONDITION}")
        lines.append("")
        lines.append("## 场景材料")
        lines.append("")
        lines.append(f"> {SCENE_A}")
        lines.append("")
        
        # 长度统计
        lines.append("## 输出长度统计")
        lines.append("")
        lines.append("| 条件 | 平均长度 | 最小 | 最大 |")
        lines.append("|------|----------|------|------|")
        
        for cond_key in ["control", "state_trust", "state_conflict"]:
            cond_results = [r for r in results if r["condition"] == cond_key]
            if cond_results:
                lengths = [r["length"] for r in cond_results]
                avg = sum(lengths) / len(lengths)
                lines.append(f"| {CONDITIONS[cond_key]['name']} | {avg:.0f} | {min(lengths)} | {max(lengths)} |")
        lines.append("")
        
        # 文本预览
        lines.append("## 样本预览")
        lines.append("")
        for cond_key in ["control", "state_trust", "state_conflict"]:
            cond_results = [r for r in results if r["condition"] == cond_key]
            if cond_results:
                lines.append(f"### {CONDITIONS[cond_key]['name']}")
                lines.append("")
                for i, r in enumerate(cond_results[:2]):
                    text = r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"]
                    lines.append(f"**样本 {i+1}** (长度: {r['length']}):")
                    lines.append("")
                    lines.append(f"> {text}")
                    lines.append("")
        
        return "\n".join(lines)


async def main():
    writer = Phase52Writer()
    await writer.run_all()

if __name__ == "__main__":
    asyncio.run(main())