#!/usr/bin/env python3
"""
Phase 6.3B: Robustness Benchmark with Revision Trace

对 14 个场景运行完整的 Runtime 流水线，收集指标并诊断修订失败原因。
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import httpx
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.runtime import (
    SceneAnalysis,
    RuntimeRouter,
    Compiler,
    PromptRenderer,
    RenderContext,
    LayerValidator,
    FailureAnalyzer,
    RevisionController,
    PatchCompiler,
    PatchRenderer,
    RevisionEngine,
    LayerControlTargets,
    LayerTarget,
    AnalysisSource,
    PolicyType,
    PredictionMode,
    RealizationMode,
    PolicyConfig,
)
from src.runtime.revision_controller import RevisionStrategy
from src.runtime.revision_trace import RevisionTrace
from src.runtime.revision_analyzer import RevisionAnalyzer, RevisionFailure

from scenes_benchmark import SCENES

# ============================================================
# 配置
# ============================================================

LLM_API_BASE = "http://localhost:8082/v1"
LLM_MODEL = "Qwen3-32B-Q5_K_M"
OUTPUT_DIR = Path(__file__).parent / "reports" / "phase6_3b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 场景运行器
# ============================================================

class BenchmarkRunner:
    def __init__(self):
        self.llm_api_base = LLM_API_BASE
        self.llm_model = LLM_MODEL
        self.router = RuntimeRouter()
        self.compiler = Compiler()
        self.renderer = PromptRenderer()
        self.validator = LayerValidator()
        self.analyzer = FailureAnalyzer()
        self.controller = RevisionController()
        self.patch_compiler = PatchCompiler()
        self.patch_renderer = PatchRenderer()
        self.revision_analyzer = RevisionAnalyzer()
        self.results = []
        self.revision_stats = {
            "no_change": 0,
            "minor_change": 0,
            "llm_ignored_prompt": 0,
            "patch_destroyed_other_layer": 0,
            "validator_unchanged": 0,
            "validator_regression": 0,
            "patch_success": 0,
        }
    
    async def call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                    openai_client = AsyncOpenAI(
                        api_key="not-needed",
                        base_url=self.llm_api_base,
                        http_client=client,
                    )
                    response = await openai_client.chat.completions.create(
                        model=self.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=2048,
                    )
                    content = response.choices[0].message.content or ""
                    if content.strip():
                        return content
            except Exception as e:
                print(f"    Attempt {attempt+1} failed: {e}")
            await asyncio.sleep((attempt + 1) * 2)
        return ""
    
    async def run_scene(self, scene_def: Dict) -> Dict:
        """运行单个场景"""
        scene_id = scene_def["scene_id"]
        tr = scene_def["tr"]
        scene_a = scene_def["scene_a"]
        state = scene_def["state"]
        
        print(f"\n--- {scene_id} (TR={tr:.2f}) ---")
        
        # 1. Router 决策
        analysis = SceneAnalysis(
            tr=tr,
            prediction_plasticity=1.0 - tr,
            source=AnalysisSource.MEASURED,
            confidence=0.85,
            state_type="exploration",
            reason="来自 TR 测量实验",
            features={}
        )
        decision = self.router.route(analysis)
        policy = decision.policy_config
        
        print(f"  Policy: {policy.policy_type.value} (pred={policy.prediction.value}, real={policy.realization.value})")
        
        # 2. Compiler → Renderer → Writer
        layer_targets = self.compiler.compile(policy)
        render_context = RenderContext(
            layer_targets=layer_targets,
            state=state,
            scene_analysis=analysis,
            policy=policy,
            scene_a=scene_a,
        )
        prompt = self.renderer.render(render_context)
        
        # 3. 调用 Writer (LLM)
        draft = await self.call_llm(prompt)
        print(f"  Draft length: {len(draft)}")
        
        # 4. Validator
        report_before = self.validator.validate(draft, layer_targets)
        print(f"  Draft Compliance: {report_before.compliance_rate:.2f}")
        
        # 5. 诊断
        diagnosis = self.analyzer.analyze(report_before, draft, layer_targets)
        
        # 6. Controller
        revision_decision = self.controller.decide(diagnosis, report_before, None)
        
        # 7. 执行修订（如果需要）
        final_text = draft
        patch_plan = None
        revision_result = None
        report_after = report_before
        revision_trace = None
        revision_diagnosis = None
        
        if revision_decision.should_revise and revision_decision.strategy != RevisionStrategy.SKIP:
            print(f"  Revision: {revision_decision.strategy.value}")
            
            # 生成 PatchPlan
            patch_plan = self.patch_compiler.compile(revision_decision, diagnosis)
            
            # 渲染 Patch
            patch_prompt = self.patch_renderer.render(patch_plan, draft)
            
            # 执行修订
            revision_engine = RevisionEngine(LLM_API_BASE, LLM_MODEL)
            revision_result = await revision_engine.revise(draft, patch_plan)
            final_text = revision_result.patched_text
            
            # 第二轮验证
            report_after = self.validator.validate(final_text, layer_targets)
            
            # 构建 RevisionTrace
            import time as time_module
            revision_trace = RevisionTrace(
                original_text=draft,
                patch_plan=patch_plan,
                patch_prompt=patch_prompt,
                revised_text=final_text,
                diff=self._compute_diff(draft, final_text),
                token_count_before=len(draft),
                token_count_after=len(final_text),
                validator_before=report_before,
                validator_after=report_after,
                revision_strategy=revision_decision.strategy.value,
                target_layers=revision_decision.target_layers,
                timestamp=time_module.time(),
            )
            
            # 分析修订
            revision_diagnosis = self.revision_analyzer.analyze(revision_trace)
            
            # 统计
            self.revision_stats[revision_diagnosis.failure_type.value] += 1
            
            print(f"  Final Compliance: {report_after.compliance_rate:.2f}")
            print(f"  Revision Diagnosis: {revision_diagnosis.failure_type.value}")
        else:
            print(f"  Revision: SKIP")
        
        # 8. 记录结果
        result = {
            "scene_id": scene_id,
            "tr": tr,
            "policy": policy.policy_type.value,
            "prediction_mode": policy.prediction.value,
            "realization_mode": policy.realization.value,
            "layer_targets": layer_targets.to_dict(),
            "draft": draft,
            "draft_compliance": {
                "rate": report_before.compliance_rate,
                "prediction": report_before.prediction.compliant,
                "reasoning": report_before.reasoning.compliant,
                "justification": report_before.justification.compliant,
                "construction": report_before.construction.compliant,
            },
            "diagnosis": {
                "analyses": [a.to_dict() for a in diagnosis.analyses] if diagnosis.analyses else [],
                "requires_attention": diagnosis.requires_attention,
            },
            "revision_decision": {
                "should_revise": revision_decision.should_revise,
                "strategy": revision_decision.strategy.value,
                "target_layers": revision_decision.target_layers,
                "rationale": revision_decision.rationale,
            },
            "patch_plan": {
                "actions": [
                    {"layer": a.layer, "operation": a.operation.value}
                    for a in patch_plan.actions
                ] if patch_plan else [],
                "revision_required": patch_plan.revision_required if patch_plan else False,
            } if patch_plan else None,
            "revision_result": {
                "modified": revision_result.modified if revision_result else False,
                "actions_applied": revision_result.actions_applied if revision_result else 0,
            } if revision_result else None,
            "final_text": final_text,
            "final_compliance": {
                "rate": report_after.compliance_rate,
                "prediction": report_after.prediction.compliant,
                "reasoning": report_after.reasoning.compliant,
                "justification": report_after.justification.compliant,
                "construction": report_after.construction.compliant,
            },
            "compliance_improved": report_after.compliance_rate > report_before.compliance_rate,
            "revision_trace": {
                "has_change": revision_trace.has_change if revision_trace else False,
                "diff_summary": revision_trace.diff_summary if revision_trace else "",
                "target_layers": revision_trace.target_layers if revision_trace else [],
                "strategy": revision_trace.revision_strategy if revision_trace else "",
            } if revision_trace else None,
            "revision_diagnosis": {
                "failure_type": revision_diagnosis.failure_type.value if revision_diagnosis else "no_revision",
                "confidence": revision_diagnosis.confidence if revision_diagnosis else 1.0,
                "evidence": revision_diagnosis.evidence if revision_diagnosis else [],
                "recommendation": revision_diagnosis.recommendation if revision_diagnosis else "",
            } if revision_diagnosis else None,
        }
        
        return result
    
    def _compute_diff(self, original: str, revised: str) -> str:
        """计算文本差异摘要"""
        if original == revised:
            return "无变化"
        added = len(revised) - len(original)
        return f"长度变化: {added:+d} 字符"
    
    async def run_all(self):
        """运行所有场景"""
        print("=" * 60)
        print("Phase 6.3B: Robustness Benchmark (with Revision Trace)")
        print("=" * 60)
        print(f"Scenes: {len(SCENES)}")
        print(f"Model: {LLM_MODEL}")
        print("=" * 60)
        
        for scene_def in SCENES:
            result = await self.run_scene(scene_def)
            self.results.append(result)
            await asyncio.sleep(2.0)
        
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """保存结果"""
        output_path = OUTPUT_DIR / "benchmark_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n结果已保存到: {output_path}")
    
    def print_summary(self):
        """打印摘要"""
        print("\n" + "=" * 60)
        print("Benchmark 摘要")
        print("=" * 60)
        
        n = len(self.results)
        draft_rates = [r["draft_compliance"]["rate"] for r in self.results]
        final_rates = [r["final_compliance"]["rate"] for r in self.results]
        improved = sum(1 for r in self.results if r["compliance_improved"])
        retries = sum(1 for r in self.results if r["revision_decision"]["strategy"] == "full_retry")
        
        print(f"\n总场景数: {n}")
        print(f"Draft Compliance (平均): {sum(draft_rates)/n:.3f}")
        print(f"Final Compliance (平均): {sum(final_rates)/n:.3f}")
        print(f"改进场景数: {improved}/{n} ({improved/n*100:.1f}%)")
        print(f"完全重试率: {retries}/{n} ({retries/n*100:.1f}%)")
        
        print("\n修订失败分布:")
        total = sum(self.revision_stats.values())
        for failure_type, count in self.revision_stats.items():
            if count > 0:
                print(f"  {failure_type}: {count}/{total} ({count/total*100:.1f}%)")
        
        print("\n各场景详情:")
        print("| 场景 | TR | Draft | Final | 改进 | 策略 | 修订诊断 |")
        print("|------|----|-------|-------|------|------|---------|")
        for r in self.results:
            diag = r.get("revision_diagnosis", {})
            diag_type = diag.get("failure_type", "N/A") if diag else "N/A"
            print(f"| {r['scene_id']} | {r['tr']:.2f} | {r['draft_compliance']['rate']:.2f} | {r['final_compliance']['rate']:.2f} | {'✓' if r['compliance_improved'] else '✗'} | {r['revision_decision']['strategy']} | {diag_type} |")


async def main():
    runner = BenchmarkRunner()
    await runner.run_all()

if __name__ == "__main__":
    asyncio.run(main())