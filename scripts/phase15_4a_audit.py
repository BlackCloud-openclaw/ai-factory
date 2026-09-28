#!/usr/bin/env python
"""
Phase 15.4A — Contract Compliance Audit

离线分析 210 个 Shadow 样本，找出 Writer → Validator FAIL 的构成。
不修改任何 Runtime 代码。
"""

import sys
import asyncio
import asyncpg
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.writing.planning_contract import PlanningContract, SignalSource
from src.writing.validation.semantic_validator import SemanticValidator
from src.writing.state_change_types import StateChangeType

DSN = "postgresql://woami:kali@localhost:5432/ai_factory"


def is_malformed_state_change(sc) -> bool:
    """检查 StateChange 是否格式错误（type 无效或 source=UNKNOWN）"""
    # 检查 type
    valid_types = StateChangeType.values()
    if sc.get("type") not in valid_types:
        return True
    # 检查 source
    source = sc.get("source")
    if source in ("unknown", SignalSource.UNKNOWN.value) or source is None:
        return True
    # 检查 confidence（可选）
    if sc.get("confidence", 0.0) < 0.0 or sc.get("confidence", 0.0) > 1.0:
        return True
    return False


async def audit():
    conn = await asyncpg.connect(DSN)
    rows = await conn.fetch("""
        SELECT 
            scene_id,
            contract_data,
            original_text,
            original_passed
        FROM shadow_rewrite_log
        WHERE experiment_id = 'phase15.3.v1'
          AND status IN ('success', 'validation_failed')
          AND contract_data IS NOT NULL
    """)
    await conn.close()

    total = len(rows)
    print(f"📊 总样本: {total}")
    pass_count = sum(1 for r in rows if r["original_passed"])
    fail_count = total - pass_count
    print(f"   PASS: {pass_count} ({pass_count/total*100:.2f}%)")
    print(f"   FAIL: {fail_count} ({fail_count/total*100:.2f}%)")
    print("=" * 60)

    # 初始化 SemanticValidator
    validator = SemanticValidator()

    # 统计
    malformed_count = 0
    writer_missing_count = 0
    partial_count = 0
    other_count = 0

    missing_type_counts = defaultdict(int)
    total_missing = 0
    total_requirements = 0

    for row in rows:
        if row["original_passed"]:
            continue  # 跳过 PASS 样本
        
        scene_id = row["scene_id"]
        contract_dict = json.loads(row["contract_data"]) if isinstance(row["contract_data"], str) else row["contract_data"]
        original_text = row["original_text"]
        
        try:
            contract = PlanningContract(**contract_dict)
        except Exception as e:
            malformed_count += 1
            print(f"⚠️  Malformed Contract for {scene_id}: {e}")
            continue

        # 检查每个 StateChange 是否格式错误
        has_malformed = False
        for sc in contract.observables.state_changes:
            if is_malformed_state_change(sc.model_dump()):
                has_malformed = True
                break
        if has_malformed:
            malformed_count += 1
            continue

        # 有效的 Contract，用 SemanticValidator 验证
        result = validator.validate(contract, original_text)
        missing = result.missing
        missing_count = len(missing)
        total_requirements += len(contract.observables.state_changes)
        
        if missing_count == 0:
            # Contract 有效，Writer 全部实现，但 Validator 仍判 FAIL（其他原因）
            other_count += 1
        elif missing_count == len(contract.observables.state_changes):
            # Writer 完全没有实现任何要求
            writer_missing_count += 1
        else:
            # 部分实现
            partial_count += 1

        # 统计缺失类型
        for m in missing:
            # 尝试从缺失项中提取类型
            # missing 是 list of str，但通常包含场景描述
            # 简单启发式：检查常见的 StateChange.type
            found = False
            for sc in contract.observables.state_changes:
                if sc.name and sc.name in m:
                    missing_type_counts[sc.type] += 1
                    found = True
                    break
                elif sc.type and sc.type in m:
                    missing_type_counts[sc.type] += 1
                    found = True
                    break
            if not found:
                # 如果无法匹配，归入 "other"
                missing_type_counts["other"] += 1
            total_missing += 1

    # 输出统计
    print("\n📋 失败构成:")
    print(f"   Malformed Contract: {malformed_count} ({malformed_count/fail_count*100:.2f}%)")
    print(f"   Writer completely missing: {writer_missing_count} ({writer_missing_count/fail_count*100:.2f}%)")
    print(f"   Partial compliance: {partial_count} ({partial_count/fail_count*100:.2f}%)")
    print(f"   Other Validator failures: {other_count} ({other_count/fail_count*100:.2f}%)")
    print()
    print("📈 缺失类型 Top-10 (仅统计有效 Contract):")
    sorted_missing = sorted(missing_type_counts.items(), key=lambda x: x[1], reverse=True)
    for typ, count in sorted_missing[:10]:
        print(f"   {typ}: {count} ({count/total_missing*100:.2f}%)")

    print("\n✅ 审计完成")

if __name__ == "__main__":
    asyncio.run(audit())