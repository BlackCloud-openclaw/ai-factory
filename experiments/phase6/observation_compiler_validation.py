"""
Phase 6.3C-1: ObservationCompiler Validation

验证 ObservationCompiler 在全部 14 个场景上的稳定性和结构完整性。

运行方式:
    cd /home/data/projects/ai_factory
    python -m experiments.phase6.observation_compiler_validation
"""

import json
import os
import sys
from typing import Dict, List, Any

# 确保项目根目录在 sys.path 中（以便导入 src.runtime）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.runtime.observation_compiler import ObservationCompiler, ObservationIR


# ---------- 1. 加载场景 Draft ----------
BENCHMARK_FILE = os.path.join(
    PROJECT_ROOT,
    "experiments/phase6/reports/phase6_3b/benchmark_results.json"
)


def load_drafts_from_benchmark() -> Dict[str, str]:
    """从 Phase 6.3B 基准结果中提取场景 draft"""
    drafts = {}
    if os.path.exists(BENCHMARK_FILE):
        with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                scene_id = item.get("scene_id")
                draft = item.get("draft")
                if scene_id and draft:
                    drafts[scene_id] = draft
    else:
        print(f"⚠️ 警告: 找不到 {BENCHMARK_FILE}，使用 fallback 数据（仅 reunion_low）。")
        # Fallback: 仅包含 reunion_low（已通过测试）
        drafts["reunion_low"] = """林逸的指节在袖中捏紧又松开三次。那封泛黄的密信在怀中发烫，墨迹晕染处仍能辨认出"天机阁地底第七重"的批注——正是师兄失踪前夜用特殊药水写下的。此刻那道背影肩宽腰窄，与十年前领他入门时身形分毫不差，连负手而立的习惯都未曾改变。

"你留下的信里说'血色月华'是假象。"林逸听见自己嗓音发涩，"可那天值夜的弟子明明看见红月亮贯穿三更天。"风卷起几片零落桃花，落在那人玄色衣摆上。他忽然想起师兄被除名那日，宗门大殿外也是这样漫天花雨。

背影微微颤动，却在转身时稳稳化作平静笑意："你倒是学会了观云手。"师兄指尖轻点腰间玉牌，正是已绝传的青鸾纹饰。林逸瞳孔骤缩——这枚玉牌本该与宗门禁书同毁于十年前那场大火。

月光突然刺破云层，照得青鸾纹上斑驳血迹历历可数。林逸后退半步，掌心贴住怀中密信："当年在藏经阁，是二十七个长老同时违背了'无相戒律'，对吗？"""
    return drafts


# ---------- 2. 单个 ObservationIR 验证 ----------
def validate_observation_ir(ir: ObservationIR, draft: str) -> Dict[str, Any]:
    """对单个 ObservationIR 进行结构完整性检查"""
    issues = []
    draft_len = len(draft)

    # 2.1 边界检查
    for sent in ir.sentences:
        if not (0 <= sent.start <= sent.end <= draft_len):
            issues.append(f"Sentence {sent.id} 超出边界: {sent.start}-{sent.end} vs {draft_len}")
        if sent.text != draft[sent.start:sent.end]:
            issues.append(f"Sentence {sent.id} 文本与 draft 不匹配")

    # 2.2 sentence_id 一致性
    sent_ids = {s.id for s in ir.sentences}
    for p in ir.patterns:
        if p.sentence_id not in sent_ids:
            issues.append(f"Pattern {p.id} 引用了无效的 sentence_id: {p.sentence_id}")

    # 2.3 确定性：Patterns 按 start 排序
    starts = [p.start for p in ir.patterns]
    if starts != sorted(starts):
        issues.append("Patterns 未按 start 排序")

    return {
        "issues": issues,
        "sentence_count": len(ir.sentences),
        "pattern_count": len(ir.patterns),
        "pattern_types": list(set(p.pattern_type for p in ir.patterns)),
        "is_valid": len(issues) == 0,
        "source_hash": ir.source_hash[:16],
    }


# ---------- 3. 主验证流程 ----------
def run_validation():
    print("=" * 70)
    print("Phase 6.3C-1: ObservationCompiler Validation")
    print("=" * 70)

    drafts = load_drafts_from_benchmark()
    if not drafts:
        print("❌ 错误: 没有加载到任何场景 draft。请检查基准结果文件路径。")
        return

    compiler = ObservationCompiler()
    results = {}
    total_issues = 0

    for scene_id, draft in drafts.items():
        print(f"\n[编译] {scene_id} (长度: {len(draft)})")

        try:
            ir = compiler.compile(draft)
            report = validate_observation_ir(ir, draft)

            results[scene_id] = {
                "hash": ir.source_hash[:16],
                "sentences": len(ir.sentences),
                "patterns": len(ir.patterns),
                "pattern_types": report["pattern_types"],
                "valid": report["is_valid"],
                "issues": report["issues"],
            }

            if report["is_valid"]:
                print(f"  ✅ 结构完整 (Sentences: {len(ir.sentences)}, Patterns: {len(ir.patterns)})")
                print(f"     Pattern Types: {report['pattern_types']}")
            else:
                print(f"  ❌ 发现 {len(report['issues'])} 个问题")
                for issue in report["issues"][:3]:
                    print(f"     - {issue}")
                total_issues += len(report["issues"])

        except Exception as e:
            results[scene_id] = {"valid": False, "issues": [f"异常: {str(e)}"]}
            print(f"  ❌ 编译异常: {e}")
            total_issues += 1

    # ---------- 汇总 ----------
    print("\n" + "=" * 70)
    print("验证汇总")
    print("=" * 70)
    valid_count = sum(1 for r in results.values() if r.get("valid", False))
    total_count = len(results)
    print(f"总场景数: {total_count}")
    print(f"结构完整: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    print(f"发现的问题总数: {total_issues}")

    if total_issues == 0:
        print("\n✅ 所有场景 ObservationCompiler 编译通过，结构完整！")
        print("   可以进入 Phase 6.3C-2 (ExecutionIR Benchmark)。")
    else:
        print("\n⚠️ 部分场景存在结构性问题，请在进入下一阶段前修复。")

    # 保存详细结果
    output_dir = os.path.join(PROJECT_ROOT, "experiments/phase6/reports/phase6_3c")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "observation_compiler_validation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    run_validation()
