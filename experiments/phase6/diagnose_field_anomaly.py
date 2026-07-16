"""
诊断 field_anomaly 场景的 Draft v2 结构
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.runtime.observation_compiler import ObservationCompiler
from src.runtime.validator import Validator


# 加载 Benchmark 结果
BENCHMARK_FILE = os.path.join(
    PROJECT_ROOT,
    "experiments/phase6/reports/phase6_3c/benchmark_results.json"
)

with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# 找到 field_anomaly 的结果
field_anomaly = None
for r in data.get("results", []):
    if r.get("scene_id") == "field_anomaly":
        field_anomaly = r
        break

if not field_anomaly:
    print("❌ 找不到 field_anomaly 的结果")
    sys.exit(1)

# 获取 Draft v1 和 v2（Benchmark 结果中只保存了长度，没有保存完整文本）
# 我们需要从 Phase 6.3B 的原始数据中加载
BENCHMARK_3B_FILE = os.path.join(
    PROJECT_ROOT,
    "experiments/phase6/reports/phase6_3b/benchmark_results.json"
)

with open(BENCHMARK_3B_FILE, "r", encoding="utf-8") as f:
    data_3b = json.load(f)

draft_v1 = None
for item in data_3b:
    if item.get("scene_id") == "field_anomaly":
        draft_v1 = item.get("draft")
        break

if not draft_v1:
    print("❌ 找不到 field_anomaly 的 Draft v1")
    sys.exit(1)

# 由于 Benchmark 结果没有保存 Draft v2，我们需要重新模拟执行
# 使用 Mock LLM 执行器重新生成 Draft v2
from experiments.phase6.phase6_3c_benchmark import LLMExecutor
from src.runtime.edit_compiler import EditCompiler
from src.runtime.patch_renderer import PatchRenderer

print("=" * 70)
print("诊断: field_anomaly")
print("=" * 70)

# 1. 编译 v1
compiler = ObservationCompiler()
ir_v1 = compiler.compile(draft_v1)

# 2. 验证 v1
validator = Validator()
layer_targets = {
    "reasoning": "enhanced",
    "justification": "enhanced",
    "construction": "enhanced",
    "prediction": "enhanced"
}
report_v1 = validator.validate(ir_v1, layer_targets)

print(f"\n[V1] Compliance: {report_v1.overall_compliance:.2f}")
print(f"[V1] Sentences: {len(ir_v1.sentences)}, Patterns: {len(ir_v1.patterns)}")

# 3. 生成 EditPlan
edit_compiler = EditCompiler()
plan = edit_compiler.compile(ir_v1, report_v1, diagnosis_id="D_field_anomaly")

print(f"\n[EditPlan] Actions: {len(plan.actions)}")
for action in plan.actions:
    print(f"  - {action.operation.value}: anchor={action.anchor_sentence_id}, payload={action.payload_type}")

# 4. 渲染 Patch
renderer = PatchRenderer()
rendered = renderer.render(plan, ir_v1)

# 5. 模拟执行
executor = LLMExecutor(mode="mock")
draft_v2 = executor.execute(rendered.full_prompt, draft_v1)

print(f"\n[Draft变化] {len(draft_v1)} → {len(draft_v2)} 字符")

# 6. 编译 v2
ir_v2 = compiler.compile(draft_v2)

# 7. 验证 v2
report_v2 = validator.validate(ir_v2, layer_targets)

print(f"\n[V2] Compliance: {report_v2.overall_compliance:.2f}")
print(f"[V2] Sentences: {len(ir_v2.sentences)}, Patterns: {len(ir_v2.patterns)}")

# 8. 详细对比：找出新增的句子
print("\n" + "=" * 70)
print("新增/变化的句子分析")
print("=" * 70)

# 找到 v2 中新增的句子（长度不同）
for i, s2 in enumerate(ir_v2.sentences):
    # 检查这个句子是否在 v1 中存在（通过文本匹配）
    found = False
    for s1 in ir_v1.sentences:
        if s1.text == s2.text:
            found = True
            break
    if not found:
        print(f"\n新增句子 [{s2.id}]: {s2.text[:80]}...")
        # 检查这个句子的 Patterns
        patterns = [p for p in ir_v2.patterns if p.sentence_id == s2.id]
        if patterns:
            print(f"  Patterns: {[(p.pattern_type, p.text) for p in patterns]}")
        else:
            print(f"  ❌ 该句子没有任何 Pattern！")

# 9. 检查 Validator 的判定细节
print("\n" + "=" * 70)
print("Validator 层判定详情 (v2)")
print("=" * 70)

for res in report_v2.layer_results:
    print(f"\n{res.layer}: {'✅ 合规' if res.compliant else '❌ 不合规'} (target={res.target_level})")
    if not res.compliant:
        for ev in res.evidence_list:
            # 查看该句子的文本
            anchor_sent = None
            for s in ir_v2.sentences:
                if s.id == ev.anchor_sentence_id:
                    anchor_sent = s
                    break
            if anchor_sent:
                print(f"  锚点句子: {anchor_sent.text[:60]}...")
            print(f"  缺失类型: {ev.missing_pattern_types}")
            print(f"  已有 Patterns: {ev.present_patterns}")