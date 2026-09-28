#!/usr/bin/env python
"""
Phase 15.4C Final Audit

验证 Contract Signal Reinforcement 对 Writer Compliance 的影响。

L1: Writer Compliance (Completely Missing / Partial / Full)
L2: Validator Outcome (PASS / both_pass / rescue / regression)
L3: Regression Safety (descriptive only)
"""

import sys
import asyncio
import asyncpg
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.writing.planning_contract import PlanningContract
from src.writing.validation.semantic_validator import SemanticValidator
from src.writing.state_change_types import StateChangeType

DSN = "postgresql://woami:kali@localhost:5432/ai_factory"


def classify_compliance(contract: PlanningContract, result) -> str:
    """
    对 Writer 的 Contract Compliance 进行分类。

    Full: 所有 state_changes 都被实现
    Partial: 部分实现
    Completely Missing: 完全没有实现
    """
    total = len(contract.observables.state_changes)
    if total == 0:
        return "no_requirements"

    missing_count = len(result.missing)
    if missing_count == 0:
        return "full"
    elif missing_count == total:
        return "completely_missing"
    else:
        return "partial"


async def audit():
    conn = await asyncpg.connect(DSN)

    # ============================================================
    # 1. 读取数据
    # ============================================================
    rows = await conn.fetch("""
        SELECT
            scene_id,
            contract_data,
            original_text,
            original_passed,
            rewritten_passed,
            executed_at
        FROM shadow_rewrite_log
        WHERE experiment_id = 'phase15.3.v1'
          AND status IN ('success', 'validation_failed')
          AND contract_data IS NOT NULL
    """)
    await conn.close()

    # 划分 Baseline / Treatment
    baseline_cutoff = datetime(2026, 8, 21, 0, 0, 0)
    baseline_samples = []
    treatment_samples = []

    for r in rows:
        if r["executed_at"] < baseline_cutoff:
            baseline_samples.append(r)
        else:
            treatment_samples.append(r)

    print("=" * 80)
    print("Phase 15.4C Final Audit")
    print("=" * 80)
    print(f"Baseline:  {len(baseline_samples)} samples")
    print(f"Treatment: {len(treatment_samples)} samples")
    print()

    # ============================================================
    # 2. L1: Writer Compliance
    # ============================================================
    validator = SemanticValidator()

    def analyze_compliance(samples, label):
        stats = {
            "completely_missing": 0,
            "partial": 0,
            "full": 0,
            "no_requirements": 0,
            "malformed": 0,
        }
        details = []

        for row in samples:
            scene_id = row["scene_id"]
            try:
                contract_dict = json.loads(row["contract_data"]) if isinstance(row["contract_data"], str) else row["contract_data"]
                contract = PlanningContract(**contract_dict)
            except Exception:
                stats["malformed"] += 1
                continue

            if not contract.observables.state_changes:
                stats["no_requirements"] += 1
                continue

            result = validator.validate(contract, row["original_text"])
            classification = classify_compliance(contract, result)
            stats[classification] += 1
            details.append({
                "scene_id": scene_id,
                "classification": classification,
                "total": len(contract.observables.state_changes),
                "missing": len(result.missing),
            })

        return stats, details

    baseline_stats, baseline_details = analyze_compliance(baseline_samples, "Baseline")
    treatment_stats, treatment_details = analyze_compliance(treatment_samples, "Treatment")

    print("=" * 80)
    print("L1: Writer Compliance")
    print("=" * 80)
    print()

    def print_stats(stats, label):
        total = sum(stats.values())
        print(f"  {label}:")
        print(f"    Completely Missing: {stats['completely_missing']} ({stats['completely_missing']/total*100:.2f}%)")
        print(f"    Partial:            {stats['partial']} ({stats['partial']/total*100:.2f}%)")
        print(f"    Full:               {stats['full']} ({stats['full']/total*100:.2f}%)")
        print(f"    No Requirements:    {stats['no_requirements']} ({stats['no_requirements']/total*100:.2f}%)")
        print(f"    Malformed:          {stats['malformed']} ({stats['malformed']/total*100:.2f}%)")
        print()

    print_stats(baseline_stats, "Baseline")
    print_stats(treatment_stats, "Treatment")

    # 计算 Completely Missing 率的变化
    baseline_total = sum(baseline_stats.values())
    treatment_total = sum(treatment_stats.values())
    baseline_cm = baseline_stats["completely_missing"] / max(1, baseline_total - baseline_stats["malformed"] - baseline_stats["no_requirements"]) * 100
    treatment_cm = treatment_stats["completely_missing"] / max(1, treatment_total - treatment_stats["malformed"] - treatment_stats["no_requirements"]) * 100

    print(f"  Completely Missing Rate:")
    print(f"    Baseline:  {baseline_cm:.2f}%")
    print(f"    Treatment: {treatment_cm:.2f}%")
    print(f"    Change:    {treatment_cm - baseline_cm:+.2f} 个百分点")
    print(f"    Relative:  {(treatment_cm - baseline_cm) / baseline_cm * 100:+.2f}%")
    print()

    # ============================================================
    # 3. L2: Validator Outcome
    # ============================================================
    print("=" * 80)
    print("L2: Validator Outcome")
    print("=" * 80)
    print()

    baseline_pass = sum(1 for r in baseline_samples if r["original_passed"])
    treatment_pass = sum(1 for r in treatment_samples if r["original_passed"])

    baseline_both_pass = sum(1 for r in baseline_samples if r["original_passed"] and r["rewritten_passed"])
    treatment_both_pass = sum(1 for r in treatment_samples if r["original_passed"] and r["rewritten_passed"])

    baseline_regression = sum(1 for r in baseline_samples if r["original_passed"] and not r["rewritten_passed"])
    treatment_regression = sum(1 for r in treatment_samples if r["original_passed"] and not r["rewritten_passed"])

    baseline_rescue = sum(1 for r in baseline_samples if not r["original_passed"] and r["rewritten_passed"])
    treatment_rescue = sum(1 for r in treatment_samples if not r["original_passed"] and r["rewritten_passed"])

    baseline_both_fail = sum(1 for r in baseline_samples if not r["original_passed"] and not r["rewritten_passed"])
    treatment_both_fail = sum(1 for r in treatment_samples if not r["original_passed"] and not r["rewritten_passed"])

    print("  Baseline:")
    print(f"    Original PASS:  {baseline_pass} ({baseline_pass/len(baseline_samples)*100:.2f}%)")
    print(f"    both_pass:      {baseline_both_pass}")
    print(f"    rescue:         {baseline_rescue}")
    print(f"    regression:     {baseline_regression}")
    print(f"    both_fail:      {baseline_both_fail}")
    print()

    print("  Treatment:")
    print(f"    Original PASS:  {treatment_pass} ({treatment_pass/len(treatment_samples)*100:.2f}%)")
    print(f"    both_pass:      {treatment_both_pass}")
    print(f"    rescue:         {treatment_rescue}")
    print(f"    regression:     {treatment_regression}")
    print(f"    both_fail:      {treatment_both_fail}")
    print()

    print(f"  PASS Rate Change:  {baseline_pass/len(baseline_samples)*100:.2f}% → {treatment_pass/len(treatment_samples)*100:.2f}%")
    print(f"  Relative Lift:    {(treatment_pass/len(treatment_samples) - baseline_pass/len(baseline_samples)) / (baseline_pass/len(baseline_samples)) * 100:+.2f}%")

    # ============================================================
    # 4. L3: Regression Safety
    # ============================================================
    print()
    print("=" * 80)
    print("L3: Regression Safety")
    print("=" * 80)
    print()

    total_original_pass = baseline_pass + treatment_pass
    total_regression = baseline_regression + treatment_regression

    print(f"  Total Original PASS: {total_original_pass}")
    print(f"  Total Regression:    {total_regression}")
    if total_original_pass > 0:
        print(f"  Conditional Rate:    {total_regression / total_original_pass * 100:.2f}%")
    print()
    print("  ⚠️  Due to extremely small denominator (5 Original PASS cases),")
    print("     this rate is descriptive only and insufficient to establish")
    print("     systematic regression risk.")

    # ============================================================
    # 5. Final Decision
    # ============================================================
    print()
    print("=" * 80)
    print("Final H1 Decision")
    print("=" * 80)
    print()

    # 预设阈值: Completely Missing 下降 >10%
    cm_change_relative = (treatment_cm - baseline_cm) / baseline_cm * 100

    if cm_change_relative < -10:
        print("✅ H1 SUPPORTED: Completely Missing rate decreased by >10%")
        print(f"   ({baseline_cm:.2f}% → {treatment_cm:.2f}%, relative change: {cm_change_relative:+.2f}%)")
    elif cm_change_relative < -5:
        print("🟡 H1 WEAKLY SUPPORTED: Completely Missing rate decreased by 5-10%")
        print(f"   ({baseline_cm:.2f}% → {treatment_cm:.2f}%, relative change: {cm_change_relative:+.2f}%)")
        print("   Evidence of effect, but not meeting the >10% threshold.")
    else:
        print("❌ H1 NOT SUPPORTED: Completely Missing rate did not decrease by >10%")
        print(f"   ({baseline_cm:.2f}% → {treatment_cm:.2f}%, relative change: {cm_change_relative:+.2f}%)")
        print("   Contract Signal Reinforcement did not produce a substantial improvement in Writer compliance.")

    print()
    print("=" * 80)
    print("Audit Complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(audit())