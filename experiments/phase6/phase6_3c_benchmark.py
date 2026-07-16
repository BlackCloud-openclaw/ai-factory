"""
Phase 6.3C-2 / 6.3D: ExecutionIR Benchmark + Closed-loop Stability Validation

支持：
- 单轮 Benchmark（原有功能）
- 多轮闭环验证（--rounds N）
- 幂等性、确定性、收敛性分析
- Runtime Stability Score (RS)

Phase 7 兼容性修复：
- ObservationCompiler.compile() 现在需要 snapshot 参数
- Validator.validate() 现在需要 (snapshot, ir) 而非 (ir, layer_targets)
- EditCompiler.compile() 已被 compile_with_snapshot() 取代
- 使用默认 Snapshot（仅包含 ReasoningSurface）适配 Phase 6 测试
"""

import json
import os
import sys
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.runtime.observation_compiler import ObservationCompiler
from src.runtime.validator import Validator
from src.runtime.edit_compiler import EditCompiler
from src.runtime.patch_renderer import PatchRenderer
from src.runtime.snapshot import RuntimeConfig
from src.runtime.builder import RuntimeBuilder
from src.runtime.registry import SurfaceRegistry
from src.surfaces.reasoning import ReasoningSurface


# ============================================================
# 0. 辅助函数：获取默认 Snapshot（Phase 6 兼容）
# ============================================================

def _get_default_snapshot():
    """创建默认 Snapshot（仅包含 ReasoningSurface），用于 Phase 6 测试兼容"""
    registry = SurfaceRegistry((ReasoningSurface,))
    return RuntimeBuilder(registry).from_surfaces(registry, "reasoning")


# ============================================================
# 1. 数据加载
# ============================================================

BENCHMARK_3B_FILE = os.path.join(
    PROJECT_ROOT,
    "experiments/phase6/reports/phase6_3b/benchmark_results.json"
)


def load_scenes_from_3b() -> List[Dict[str, Any]]:
    if not os.path.exists(BENCHMARK_3B_FILE):
        raise FileNotFoundError(f"找不到 {BENCHMARK_3B_FILE}")
    with open(BENCHMARK_3B_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. LLM 执行器（Mock / Real）
# ============================================================

class LLMExecutor:
    def __init__(self, mode: str = "mock"):
        self.mode = mode
        self.call_count = 0

    def execute(self, prompt: str, draft: str) -> str:
        self.call_count += 1
        if self.mode == "mock":
            return self._mock_execute(prompt, draft)
        return self._real_execute(prompt, draft)

    def _mock_execute(self, prompt: str, draft: str) -> str:
        import re
        keyword_match = re.search(r'关键词「(.*?)」', prompt)
        if not keyword_match:
            keyword = "密信"
        else:
            keyword = keyword_match.group(1)

        if "替换为" in prompt or "REPLACE" in prompt:
            anchor_match = re.search(r'「(.*?)」', prompt)
            if anchor_match:
                anchor_text = anchor_match.group(1)
                if anchor_text in draft:
                    new_sentence = f"{anchor_text}他忽然意识到，{keyword}与眼前的一切之间存在着尚未看透的联系。"
                    return draft.replace(anchor_text, new_sentence, 1)
            return draft + f"\n他忽然意识到，{keyword}与眼前的一切之间存在着尚未看透的联系。"

        anchor_match = re.search(r'「(.*?)」', prompt)
        if not anchor_match:
            return draft + f"\n\n（插入包含「{keyword}」的内容）"

        anchor_text = anchor_match.group(1)
        search_text = anchor_text[:20]
        if search_text in draft:
            insert_text = f"\n他忽然意识到，{keyword}与眼前的一切之间存在着尚未看透的联系。"
            return draft.replace(search_text, search_text + insert_text, 1)

        return draft + f"\n\n（插入包含「{keyword}」的内容）"

    def _real_execute(self, prompt: str, draft: str) -> str:
        raise NotImplementedError("真实 LLM 调用尚未实现")


# ============================================================
# 3. 核心测试函数：单轮
# ============================================================

def run_single_round(
    draft: str,
    layer_targets: Dict[str, str],  # 保留但不再使用
    round_num: int,
    llm_executor: LLMExecutor,
    verbose: bool = True,
    force_edit: bool = False
) -> Dict[str, Any]:
    """执行单轮修订"""
    obs_compiler = ObservationCompiler()
    validator = Validator()
    edit_compiler = EditCompiler()
    renderer = PatchRenderer()
    
    snapshot = _get_default_snapshot()

    # 编译
    ir = obs_compiler.compile(draft, snapshot)
    report = validator.validate(snapshot, ir)  # Phase 7 新签名

    if report.overall_compliance >= 1.0 and not force_edit:
        return {
            "draft": draft,
            "compliance": report.overall_compliance,
            "layer_compliance": {r.layer: r.compliant for r in report.layer_results},
            "changed": False,
            "actions": 0,
            "ir_hash": ir.source_hash,
            "sentences": len(ir.sentences),
            "patterns": len(ir.patterns)
        }

    # 使用新签名 compile_with_snapshot
    plan = edit_compiler.compile_with_snapshot(
        snapshot, report, draft, ir, diagnosis_id=f"D_round_{round_num}"
    )
    if not plan.actions:
        return {
            "draft": draft,
            "compliance": report.overall_compliance,
            "layer_compliance": {r.layer: r.compliant for r in report.layer_results},
            "changed": False,
            "actions": 0,
            "ir_hash": ir.source_hash,
            "sentences": len(ir.sentences),
            "patterns": len(ir.patterns)
        }

    rendered = renderer.render(plan, ir)
    new_draft = llm_executor.execute(rendered.full_prompt, draft)
    changed = (new_draft != draft)

    ir_new = obs_compiler.compile(new_draft, snapshot)
    report_new = validator.validate(snapshot, ir_new)

    return {
        "draft": new_draft,
        "compliance": report_new.overall_compliance,
        "layer_compliance": {r.layer: r.compliant for r in report_new.layer_results},
        "changed": changed,
        "actions": len(plan.actions),
        "ir_hash": ir_new.source_hash,
        "sentences": len(ir_new.sentences),
        "patterns": len(ir_new.patterns),
        "prev_draft": draft,
        "prev_compliance": report.overall_compliance,
    }


# ============================================================
# 4. 多轮测试 + 稳定性分析
# ============================================================

def run_stability_test(
    scene_data: Dict[str, Any],
    llm_executor: LLMExecutor,
    max_rounds: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    对单个场景运行多轮闭环测试，并收集稳定性指标
    """
    scene_id = scene_data.get("scene_id", "unknown")
    draft = scene_data.get("draft", "")
    layer_targets = scene_data.get("layer_targets", {})
    tr = scene_data.get("tr", 0.5)
    policy = scene_data.get("policy", "adaptive")

    rounds = []
    current_draft = draft
    converged_round = None
    compliance_trace = []
    draft_length_trace = []

    obs_compiler = ObservationCompiler()
    snapshot = _get_default_snapshot()

    # ----- 确定性测试：连续编译 3 次看 hash 是否一致 -----
    hashes = []
    for _ in range(3):
        ir = obs_compiler.compile(current_draft, snapshot)
        hashes.append(ir.source_hash)
    deterministic = (len(set(hashes)) == 1)

    for r in range(1, max_rounds + 1):
        result = run_single_round(
            draft=current_draft,
            layer_targets=layer_targets,
            round_num=r,
            llm_executor=llm_executor,
            verbose=verbose
        )
        rounds.append(result)
        compliance_trace.append(result["compliance"])
        draft_length_trace.append(len(result["draft"]))

        current_draft = result["draft"]

        if result["compliance"] >= 1.0 and not result["changed"]:
            converged_round = r
            break

    final_draft = current_draft
    final_compliance = rounds[-1]["compliance"] if rounds else 0.0

    idempotence_result = None
    if final_compliance >= 1.0:
        idle_result = run_single_round(
            draft=final_draft,
            layer_targets=layer_targets,
            round_num=max_rounds + 1,
            llm_executor=llm_executor,
            verbose=False,
            force_edit=True
        )
        idle_compliance_drop = final_compliance - idle_result["compliance"]
        idle_text_change = idle_result["changed"]
        idempotence_ok = (idle_compliance_drop <= 0.05 and not idle_text_change)
        idempotence_result = {
            "compliance_drop": idle_compliance_drop,
            "text_changed": idle_text_change,
            "ok": idempotence_ok
        }
    else:
        idempotence_result = {
            "compliance_drop": 0.0,
            "text_changed": False,
            "ok": False,
            "reason": "not_converged"
        }

    if len(rounds) >= 2:
        first = rounds[0]
        last = rounds[-1]
        sent_change_rate = abs(last["sentences"] - first["sentences"]) / max(first["sentences"], 1)
    else:
        sent_change_rate = 0.0

    reference_continuity_ok = (sent_change_rate <= 0.3)

    return {
        "scene_id": scene_id,
        "tr": tr,
        "policy": policy,
        "max_rounds": max_rounds,
        "rounds": rounds,
        "compliance_trace": compliance_trace,
        "draft_length_trace": draft_length_trace,
        "converged_round": converged_round,
        "converged": converged_round is not None,
        "deterministic": deterministic,
        "idempotence": idempotence_result,
        "sent_change_rate": sent_change_rate,
        "reference_continuity_ok": reference_continuity_ok,
        "final_compliance": rounds[-1]["compliance"] if rounds else 0.0,
        "final_draft_length": len(final_draft)
    }


# ============================================================
# 5. 主 Benchmark / Stability 入口
# ============================================================

def run_stability_benchmark(mode: str = "mock", max_rounds: int = 3, verbose: bool = True):
    print("=" * 70)
    print("Phase 6.3D: Closed-loop Stability Validation")
    print("=" * 70)
    print(f"LLM Mode: {mode}")
    print(f"Max Rounds: {max_rounds}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    scenes = load_scenes_from_3b()
    print(f"\n加载场景数: {len(scenes)}")

    llm_executor = LLMExecutor(mode=mode)
    results = []
    start_time = time.time()

    for i, scene in enumerate(scenes, 1):
        print(f"\n[{i}/{len(scenes)}] {scene.get('scene_id')}")
        result = run_stability_test(scene, llm_executor, max_rounds, verbose)
        results.append(result)

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.2f} 秒")
    print(f"LLM 调用次数: {llm_executor.call_count}")

    print("\n" + "=" * 70)
    print("稳定性验证汇总")
    print("=" * 70)

    total = len(results)
    converged_count = sum(1 for r in results if r.get("converged", False))
    deterministic_count = sum(1 for r in results if r.get("deterministic", False))
    idempotent_ok_count = sum(1 for r in results if r.get("idempotence", {}).get("ok", False))
    ref_continuity_count = sum(1 for r in results if r.get("reference_continuity_ok", False))

    final_compliance_avg = sum(r.get("final_compliance", 0) for r in results) / total

    round_dist = {}
    for r in results:
        cr = r.get("converged_round")
        if cr is not None:
            round_dist[cr] = round_dist.get(cr, 0) + 1
    round_dist_str = ", ".join([f"Round {k}: {v}" for k, v in sorted(round_dist.items())])

    print(f"总场景数: {total}")
    print(f"收敛率 (3轮内收敛至1.0): {converged_count}/{total} ({converged_count/total*100:.1f}%)")
    print(f"确定性 (编译3次hash一致): {deterministic_count}/{total} ({deterministic_count/total*100:.1f}%)")
    print(f"幂等性 (合规后再次修订无变化): {idempotent_ok_count}/{total} ({idempotent_ok_count/total*100:.1f}%)")
    print(f"引用连续性 (句子数变化率<30%): {ref_continuity_count}/{total} ({ref_continuity_count/total*100:.1f}%)")
    print(f"最终平均合规率: {final_compliance_avg:.3f}")
    print(f"收敛轮次分布: {round_dist_str}")

    convergence_rate = converged_count / total
    idempotence_rate = idempotent_ok_count / total
    determinism_rate = deterministic_count / total
    ref_continuity_rate = ref_continuity_count / total

    rs_score = (
        0.35 * convergence_rate +
        0.25 * idempotence_rate +
        0.20 * determinism_rate +
        0.20 * ref_continuity_rate
    )

    print(f"\nRuntime Stability Score (RS): {rs_score:.3f}")
    print(f"  收敛性权重(0.35): {convergence_rate:.3f}")
    print(f"  幂等性权重(0.25): {idempotence_rate:.3f}")
    print(f"  确定性权重(0.20): {determinism_rate:.3f}")
    print(f"  引用连续性权重(0.20): {ref_continuity_rate:.3f}")

    if rs_score >= 0.90:
        print("\n✅ Phase 6 稳定性验证通过 (RS >= 0.90)")
        print("   Runtime 已具备闭环稳定性，可进入 Phase 7")
    elif rs_score >= 0.70:
        print("\n⚠️ Phase 6 稳定性验证部分通过 (0.70 <= RS < 0.90)")
        print("   建议检查幂等性或引用连续性的失败场景")
    else:
        print("\n❌ Phase 6 稳定性验证未通过 (RS < 0.70)")
        print("   需要进一步优化 Runtime 组件")

    output_dir = os.path.join(PROJECT_ROOT, "experiments/phase6/reports/phase6_3d")
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "max_rounds": max_rounds,
        "total_scenes": total,
        "convergence_rate": convergence_rate,
        "idempotence_rate": idempotence_rate,
        "determinism_rate": determinism_rate,
        "ref_continuity_rate": ref_continuity_rate,
        "rs_score": rs_score,
        "final_compliance_avg": final_compliance_avg,
        "round_distribution": round_dist,
        "results": results,
        "llm_calls": llm_executor.call_count,
        "elapsed_seconds": elapsed
    }

    output_path = os.path.join(output_dir, "stability_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_path}")


# ============================================================
# 6. 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 6.3D Closed-loop Stability Validation")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock",
                        help="LLM 执行模式")
    parser.add_argument("--rounds", type=int, default=3,
                        help="最大修订轮数 (默认: 3)")
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式")

    args = parser.parse_args()

    run_stability_benchmark(mode=args.mode, max_rounds=args.rounds, verbose=not args.quiet)