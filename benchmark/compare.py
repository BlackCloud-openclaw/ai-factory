#!/usr/bin/env python
import json
import sys
from pathlib import Path

def compare(novel_id: str, tolerance: float = 0.02):
    baseline_path = Path(__file__).parent / "baseline" / f"{novel_id}.json"
    current_path = Path(__file__).parent / "current.json"
    if not baseline_path.exists():
        print(f"基线不存在: {baseline_path}")
        return False
    if not current_path.exists():
        print("未找到 current.json，请先运行 runner 生成当前状态")
        return False

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(current_path) as f:
        current = json.load(f)

    if baseline.get("schema_version") != current.get("schema_version"):
        print("⚠️ schema_version 不匹配")

    mismatches = []
    for chap_key, base_chap in baseline.get("chapters", {}).items():
        curr_chap = current.get("chapters", {}).get(chap_key)
        if curr_chap is None:
            mismatches.append(f"{chap_key}: 缺失")
            continue
        for qid, base_val in base_chap.items():
            curr_val = curr_chap.get(qid)
            if curr_val != base_val:
                if isinstance(base_val, (int, float)) and isinstance(curr_val, (int, float)):
                    if abs(curr_val - base_val) > tolerance:
                        mismatches.append(f"{chap_key} {qid}: {base_val} → {curr_val}")
                else:
                    mismatches.append(f"{chap_key} {qid}: {base_val} → {curr_val}")

    if mismatches:
        print("❌ 一致性检查失败:")
        for m in mismatches[:20]:
            print(f"  {m}")
        return False
    else:
        print("✅ 一致性检查通过")
        return True

if __name__ == "__main__":
    novel_id = sys.argv[1] if len(sys.argv) > 1 else "simple_long_novel_001"
    compare(novel_id)
