#!/usr/bin/env python
"""
Phase 15.4B — Root Cause Audit

B-1: 抽取 3 类对照样本
B-2: 检查最终 Prompt 中 Contract 的可见性
B-3: 检查 Writer 实际输出的 Events
"""

import sys
import asyncio
import asyncpg
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.writing.planning_contract import PlanningContract
from src.writing.validation.semantic_validator import SemanticValidator
from src.writing.state_change_types import StateChangeType

DSN = "postgresql://woami:kali@localhost:5432/ai_factory"
LOGS_DIR = Path("logs/llm_artifacts")


def is_completely_missing(contract, original_text):
    """检查 Writer 是否完全忽略了所有 state_changes"""
    validator = SemanticValidator()
    result = validator.validate(contract, original_text)
    total = len(contract.observables.state_changes)
    missing = len(result.missing)
    if total == 0:
        return False  # 没有要求，不算 completely missing
    return missing == total


def is_partial_compliance(contract, original_text):
    """检查 Writer 是否部分执行了 state_changes"""
    validator = SemanticValidator()
    result = validator.validate(contract, original_text)
    total = len(contract.observables.state_changes)
    missing = len(result.missing)
    if total == 0:
        return False
    return 0 < missing < total


def find_prompt_file(scene_id: str, timestamp: datetime) -> Path:
    """在 logs/llm_artifacts/ 中查找对应的 Prompt 文件"""
    # 尝试匹配时间戳 + scene_id
    pattern = f"*{scene_id}*prompt.txt"
    matches = list(LOGS_DIR.glob(pattern))
    if matches:
        return matches[0]
    
    # 如果找不到，尝试用时间戳模糊匹配
    if timestamp:
        date_str = timestamp.strftime("%Y%m%d_%H%M%S")
        pattern = f"{date_str}*prompt.txt"
        matches = list(LOGS_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def extract_events_from_prompt(prompt_text: str) -> list:
    """从 Prompt 中提取 state_changes"""
    # 查找 "observables" 或 "state_changes" 部分
    # 简单实现：查找 JSON 块中的 state_changes
    match = re.search(r'"state_changes":\s*(\[[^\]]*\])', prompt_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return []


async def audit():
    conn = await asyncpg.connect(DSN)
    rows = await conn.fetch("""
        SELECT 
            scene_id,
            contract_data,
            original_text,
            original_passed,
            executed_at
        FROM shadow_rewrite_log
        WHERE experiment_id = 'phase15.3.v1'
          AND status IN ('success', 'validation_failed')
          AND contract_data IS NOT NULL
    """)
    await conn.close()

    total = len(rows)
    pass_samples = [r for r in rows if r["original_passed"]]
    fail_samples = [r for r in rows if not r["original_passed"]]

    print("=" * 80)
    print("Phase 15.4B — Root Cause Audit")
    print("=" * 80)
    print(f"总样本: {total}")
    print(f"  PASS: {len(pass_samples)} (1 个)")
    print(f"  FAIL: {len(fail_samples)} (209 个)")
    print()

    # B-1: 分类 FAIL 样本
    validator = SemanticValidator()
    completely_missing = []
    partial = []
    malformed = []

    for r in fail_samples:
        scene_id = r["scene_id"]
        try:
            contract_dict = json.loads(r["contract_data"]) if isinstance(r["contract_data"], str) else r["contract_data"]
            contract = PlanningContract(**contract_dict)
        except Exception:
            malformed.append(scene_id)
            continue

        # 检查是否格式错误
        has_malformed = False
        for sc in contract.observables.state_changes:
            if sc.type not in StateChangeType.values():
                has_malformed = True
                break
            if sc.source.value in ("unknown", "UNKNOWN") or sc.source is None:
                has_malformed = True
                break
        if has_malformed:
            malformed.append(scene_id)
            continue

        # 检查 Writer 执行情况
        result = validator.validate(contract, r["original_text"])
        total_req = len(contract.observables.state_changes)
        missing_count = len(result.missing)

        if missing_count == total_req:
            completely_missing.append(scene_id)
        elif 0 < missing_count < total_req:
            partial.append(scene_id)

    print("B-1: 样本分类")
    print(f"  Completely Missing: {len(completely_missing)}")
    print(f"  Partial Compliance: {len(partial)}")
    print(f"  Malformed Contract: {len(malformed)}")
    print()

    # B-1: 抽取对照样本
    samples = []
    for sc_id in completely_missing[:3]:
        samples.append(("completely_missing", sc_id))
    for sc_id in partial[:3]:
        samples.append(("partial", sc_id))
    if pass_samples:
        samples.append(("pass", pass_samples[0]["scene_id"]))

    print("B-1: 对照样本")
    for category, scene_id in samples:
        print(f"  [{category}] {scene_id}")
    print()

    # B-2 & B-3: 详细分析
    print("=" * 80)
    print("B-2 & B-3: 逐样本分析")
    print("=" * 80)

    for category, scene_id in samples:
        print(f"\n{'='*60}")
        print(f"[{category.upper()}] {scene_id}")
        print("=" * 60)

        # 从样本中获取数据
        row = next((r for r in rows if r["scene_id"] == scene_id), None)
        if not row:
            print("  ❌ 样本数据未找到")
            continue

        contract_dict = json.loads(row["contract_data"]) if isinstance(row["contract_data"], str) else row["contract_data"]
        contract = PlanningContract(**contract_dict)
        original_text = row["original_text"]

        # B-2: 查找 Prompt 文件
        prompt_file = find_prompt_file(scene_id, row["executed_at"])
        prompt_found = prompt_file and prompt_file.exists()

        print(f"\nB-2: Prompt 可见性")
        print(f"  文件: {prompt_file.name if prompt_file else '未找到'}")
        
        if prompt_found:
            prompt_text = prompt_file.read_text(encoding="utf-8")
            prompt_len = len(prompt_text)
            print(f"  Prompt 长度: {prompt_len:,} 字符")

            # 检查 state_changes 是否出现在 Prompt 中
            sc_text = json.dumps([sc.model_dump() for sc in contract.observables.state_changes], ensure_ascii=False)
            # 简化检查：查找 state_change 的 type
            found_types = []
            for sc in contract.observables.state_changes:
                if sc.type in prompt_text:
                    found_types.append(sc.type)
                elif sc.name and sc.name in prompt_text:
                    found_types.append(sc.type)
            if found_types:
                print(f"  在 Prompt 中发现的 state_change 类型: {found_types}")
            else:
                print("  ⚠️  Prompt 中未发现任何 state_change 类型")

            # 检查位置
            if contract.observables.state_changes:
                first_sc = contract.observables.state_changes[0]
                if first_sc.type in prompt_text:
                    pos = prompt_text.find(first_sc.type)
                    pct = pos / prompt_len
                    print(f"  首个 state_change 位置: {pct:.1%} (前 1/3: {pct < 0.33}, 中 1/3: {0.33 <= pct < 0.66})")
        else:
            print("  ⚠️  Prompt 文件未找到，无法验证可见性")

        # B-3: Writer 实际输出的 Events
        print(f"\nB-3: Writer 实际输出")
        print(f"  Contract 要求 {len(contract.observables.state_changes)} 个 state_changes:")
        for sc in contract.observables.state_changes:
            print(f"    - {sc.type}: {sc.name or sc.actor or sc.location or '(无名称)'}")

        # 从 Shadow 样本中获取 original_text，尝试解析 events
        events = []
        # 尝试从 original_text 中提取 JSON 的 events 字段
        json_match = re.search(r'"events":\s*(\[[^\]]*\])', original_text, re.DOTALL)
        if json_match:
            try:
                events_data = json.loads(json_match.group(1))
                events = [e.get("type") for e in events_data if e.get("type")]
            except:
                pass
        
        # 如果没解析到，尝试用 SemanticValidator 的 matched 信息
        if not events:
            result = validator.validate(contract, original_text)
            if result.matched:
                events = [e.event_text for e in result.matched[:5]]
            if not events:
                events = ["(无 events 可提取)"]

        print(f"  Writer 输出的 events 类型:")
        for e in events[:5]:
            print(f"    - {e}")
        if len(events) > 5:
            print(f"    ... 共 {len(events)} 个 events")

        # 判断
        if category == "completely_missing":
            print("\n  📊 判断: Writer 未产生任何可匹配的 state_change 事件")
            if prompt_found and not any(sc.type in prompt_text for sc in contract.observables.state_changes):
                print("     → 疑似根因: Contract 信号未进入最终 Prompt")
            else:
                print("     → 疑似根因: Writer 无视了可见的 Contract 信号")
        elif category == "partial":
            print("\n  📊 判断: Writer 实现了部分 state_changes，但遗漏了其他")
            print("     → 疑似根因: Writer 对 Contract 的响应是部分的，可能受 Prompt 结构影响")
        elif category == "pass":
            print("\n  📊 判断: 唯一 PASS 样本，Writer 成功实现了所有 state_changes")
            print("     → 分析此样本的 Prompt 结构，与 FAIL 样本对比")

    print("\n" + "=" * 80)
    print("B-1~B-3 完成")
    print("=" * 80)
    print("\n⚠️  注意: 这是根因定位，不是修复。下一步根据结果决定 15.4C 方向。")


if __name__ == "__main__":
    asyncio.run(audit())