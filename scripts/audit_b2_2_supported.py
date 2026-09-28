#!/usr/bin/env python
"""
B2-2A.1: 人工校准工具

从 Replay 报告中抽取 SUPPORTED 样本，按类型分层，生成标注表。
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置随机种子以保证可复现
random.seed(42)


def load_replay_report(report_path: Path) -> dict:
    """加载 Replay 报告"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_supported_claims(report: dict) -> list:
    """提取所有 SUPPORTED 的 Claim"""
    supported = []
    
    for detail in report.get("details", []):
        sample_id = detail.get("sample_id")
        scene_id = detail.get("scene_id")
        
        for sc in detail.get("state_changes", []):
            if sc.get("verdict") == "SUPPORTED":
                supported.append({
                    "sample_id": sample_id,
                    "scene_id": scene_id,
                    "claim_id": sc.get("claim_id"),
                    "state_change_type": sc.get("state_change_type", "unknown"),
                    "verdict": sc.get("verdict"),
                    "confidence": sc.get("confidence", 0.0),
                    "reason": sc.get("reason", ""),
                    "judgement": sc.get("judgement", {}),
                })
    
    return supported


def classify_by_type(supported: list) -> dict:
    """按类型分类"""
    classified = defaultdict(list)
    
    for item in supported:
        change_type = item.get("state_change_type", "unknown")
        classified[change_type].append(item)
    
    return dict(classified)


def sample_for_audit(
    classified: dict,
    realm_sample_count: int = 10,
    non_realm_sample_count: int = 10,
) -> list:
    """分层抽样"""
    sampled = []
    
    # realm_change
    realm_items = classified.get("realm_change", [])
    if len(realm_items) >= realm_sample_count:
        sampled.extend(random.sample(realm_items, realm_sample_count))
    else:
        sampled.extend(realm_items)
        print(f"⚠️ realm_change 只有 {len(realm_items)} 个，全部抽取")
    
    # 非 realm_change
    non_realm = []
    for change_type, items in classified.items():
        if change_type != "realm_change":
            non_realm.extend(items)
    
    if len(non_realm) >= non_realm_sample_count:
        sampled.extend(random.sample(non_realm, non_realm_sample_count))
    else:
        sampled.extend(non_realm)
        print(f"⚠️ 非 realm_change 只有 {len(non_realm)} 个，全部抽取")
    
    return sampled


def generate_audit_table(sampled: list, output_path: Path):
    """生成人工标注表"""
    table = []
    
    for idx, item in enumerate(sampled, 1):
        table.append({
            "序号": idx,
            "sample_id": item.get("sample_id"),
            "scene_id": item.get("scene_id"),
            "claim_id": item.get("claim_id"),
            "类型": item.get("state_change_type"),
            "置信度": item.get("confidence", 0.0),
            "Judge 判定": item.get("verdict"),
            "Judge 理由": item.get("reason", ""),
            # 人工标注字段
            "人工判定": "",  # TRUE / FALSE / UNCERTAIN
            "人工备注": "",
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已生成标注表: {output_path}")
    print(f"   共 {len(table)} 条待标注")
    
    # 打印统计
    type_counts = {}
    for item in sampled:
        t = item.get("state_change_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("\n📊 样本类型分布:")
    for t, count in type_counts.items():
        print(f"   {t}: {count}")


def main():
    # 查找最新的 Replay 报告
    reports_dir = project_root / "reports"
    report_files = list(reports_dir.glob("b2_2_replay_*.json"))
    
    if not report_files:
        print("❌ 未找到 Replay 报告")
        return
    
    # 取最新的
    latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 使用报告: {latest_report}")
    
    # 加载报告
    report = load_replay_report(latest_report)
    
    # 提取 SUPPORTED
    supported = extract_supported_claims(report)
    print(f"📊 总 SUPPORTED 数: {len(supported)}")
    
    if not supported:
        print("❌ 没有 SUPPORTED 样本")
        return
    
    # 分类
    classified = classify_by_type(supported)
    print("\n📊 SUPPORTED 按类型分布:")
    for t, items in classified.items():
        print(f"   {t}: {len(items)}")
    
    # 抽样
    sampled = sample_for_audit(
        classified,
        realm_sample_count=10,
        non_realm_sample_count=10,
    )
    
    # 生成标注表
    output_path = reports_dir / "b2_2_audit_supported.json"
    generate_audit_table(sampled, output_path)
    
    # 同时生成一个人类可读的 Markdown 版本
    md_path = reports_dir / "b2_2_audit_supported.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# B2-2A.1 人工校准标注表\n\n")
        f.write(f"生成时间: {latest_report.stem}\n\n")
        f.write("| 序号 | sample_id | 类型 | 置信度 | Judge 理由 | 人工判定 | 备注 |\n")
        f.write("|------|-----------|------|--------|------------|----------|------|\n")
        
        for idx, item in enumerate(sampled, 1):
            reason = item.get("reason", "")[:60]
            f.write(f"| {idx} | {item.get('sample_id')} | {item.get('state_change_type')} | {item.get('confidence', 0.0):.2f} | {reason}... | | |\n")
    
    print(f"✅ 已生成 Markdown 版本: {md_path}")
    
    # 输出分布详情
    print("\n📋 抽样详情:")
    for idx, item in enumerate(sampled, 1):
        print(f"  {idx}. sample={item.get('sample_id')}, type={item.get('state_change_type')}, conf={item.get('confidence', 0.0):.2f}")


if __name__ == "__main__":
    main()