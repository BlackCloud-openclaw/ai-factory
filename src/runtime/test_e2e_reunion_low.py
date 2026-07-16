"""
Phase 6.3C - End-to-End Regression Test (reunion_low)

测试目标：
1. 验证 ObservationCompiler 能稳定生成 ObservationIR。
2. 验证 Validator 能输出基于 ID 的合规报告。
3. 验证 EditCompiler 能将诊断编译为可执行的 EditPlan。
4. 验证 PatchRenderer 能将 EditPlan 渲染为精确的 LLM 指令。
5. 验证 LLM 执行 Patch 后，Compliance 是否从 False 变为 True。
"""
import sys
import os
# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from src.runtime.observation_compiler import ObservationCompiler
from src.runtime.validator import Validator
from src.runtime.edit_compiler import EditCompiler
from src.runtime.patch_renderer import PatchRenderer


# ============================================================
# 1. 测试数据（reunion_low 场景）
# ============================================================

DRAFT_V1 = """林逸的指节在袖中捏紧又松开三次。那封泛黄的密信在怀中发烫，墨迹晕染处仍能辨认出"天机阁地底第七重"的批注——正是师兄失踪前夜用特殊药水写下的。此刻那道背影肩宽腰窄，与十年前领他入门时身形分毫不差，连负手而立的习惯都未曾改变。

"你留下的信里说'血色月华'是假象。"林逸听见自己嗓音发涩，"可那天值夜的弟子明明看见红月亮贯穿三更天。"风卷起几片零落桃花，落在那人玄色衣摆上。他忽然想起师兄被除名那日，宗门大殿外也是这样漫天花雨。

背影微微颤动，却在转身时稳稳化作平静笑意："你倒是学会了观云手。"师兄指尖轻点腰间玉牌，正是已绝传的青鸾纹饰。林逸瞳孔骤缩——这枚玉牌本该与宗门禁书同毁于十年前那场大火。

月光突然刺破云层，照得青鸾纹上斑驳血迹历历可数。林逸后退半步，掌心贴住怀中密信："当年在藏经阁，是二十七个长老同时违背了'无相戒律'，对吗？"""

LAYER_TARGETS = {
    "reasoning": "enhanced",
    "justification": "enhanced",
    "construction": "enhanced"
}


# ============================================================
# 2. 模拟 LLM 执行器（MVP 阶段使用模拟响应）
# ============================================================

class MockLLMExecutor:
    """
    模拟 LLM 执行器（用于快速验证，不依赖真实 API）。
    在真实测试中，可替换为 Qwen/OpenAI 调用。
    """
    
    def execute(self, prompt: str) -> str:
        """
        根据 Prompt 生成模拟响应。
        对于 reunion_low 场景，我们手动构造一个包含 "密信" 的插入响应。
        """
        # 简单策略：在包含 "忽然想起" 的句子后插入一段包含 "密信" 的内容
        # 实际场景中，这里会调用真实的 LLM API
        
        # 查找 "忽然想起" 的位置并插入
        insert_text = "他忽然意识到，密信上的血契印记与那道背影的负手姿态之间，存在着某种他尚未看透的联系。"
        
        if "忽然想起" in DRAFT_V1:
            # 在 "忽然想起" 所在的句子后面插入
            lines = DRAFT_V1.split('\n')
            modified_lines = []
            for line in lines:
                modified_lines.append(line)
                if "忽然想起" in line:
                    modified_lines.append(insert_text)
            return '\n'.join(modified_lines)
        
        # 如果没找到锚点，在末尾追加
        return DRAFT_V1 + "\n\n" + insert_text


# ============================================================
# 3. 端到端测试函数
# ============================================================

def run_end_to_end_test(draft_v1: str, layer_targets: dict, use_mock_llm: bool = True):
    """
    完整的单场景测试流程。
    """
    print("=" * 70)
    print("Phase 6.3C - End-to-End Regression Test (reunion_low)")
    print("=" * 70)

    # ---------- Step 3.1: 编译 ObservationIR v1 ----------
    print("\n[1] ObservationCompiler → ObservationIR v1")
    obs_compiler = ObservationCompiler()
    ir_v1 = obs_compiler.compile(draft_v1)
    
    print(f"  Source Hash: {ir_v1.source_hash[:16]}...")
    print(f"  Sentences: {len(ir_v1.sentences)}")
    print(f"  Patterns: {len(ir_v1.patterns)}")

    # ---------- Step 3.2: Validator 检查 ----------
    print("\n[2] Validator → ComplianceReport v1")
    validator = Validator()
    report_v1 = validator.validate(ir_v1, layer_targets)
    
    print(f"  Overall Compliance: {report_v1.overall_compliance:.2f}")
    for res in report_v1.layer_results:
        print(f"    {res.layer}: {'✅' if res.compliant else '❌'} (target={res.target_level})")

    # ---------- Step 3.3: EditCompiler 生成 EditPlan ----------
    print("\n[3] EditCompiler → EditPlan")
    edit_compiler = EditCompiler()
    plan = edit_compiler.compile(ir_v1, report_v1, diagnosis_id="D001_reunion_low")
    
    print(f"  Target Layers: {plan.target_layers}")
    print(f"  Actions: {len(plan.actions)}")
    for action in plan.actions:
        print(f"    {action.operation.value}: anchor={action.anchor_sentence_id}, payload_type={action.payload_type}")

    # ---------- Step 3.4: PatchRenderer 渲染 Prompt ----------
    print("\n[4] PatchRenderer → RenderedPatch")
    renderer = PatchRenderer()
    rendered = renderer.render(plan, ir_v1)
    
    print(f"  System Prompt Length: {len(rendered.system_prompt)}")
    print(f"  Edit Instructions: {len(rendered.edit_instructions)}")
    print(f"  Preserve Constraints: {len(rendered.preserve_constraints)}")
    print("\n  --- 渲染的编辑指令（前 200 字符）---")
    print(rendered.full_prompt[:200] + "...")

    # ---------- Step 3.5: 执行 Patch (LLM) ----------
    print("\n[5] Execution Backend (LLM) → Draft v2")
    if use_mock_llm:
        executor = MockLLMExecutor()
        draft_v2 = executor.execute(rendered.full_prompt)
    else:
        # 真实 LLM 调用（暂未实现）
        print("  真实 LLM 调用未实现，使用模拟响应。")
        draft_v2 = draft_v1  # fallback

    print(f"  Draft v1 Length: {len(draft_v1)}")
    print(f"  Draft v2 Length: {len(draft_v2)}")
    print(f"  Change Detected: {'✅' if draft_v1 != draft_v2 else '❌'}")

    # ---------- Step 3.6: 重新编译 ObservationIR v2 ----------
    print("\n[6] ObservationCompiler → ObservationIR v2")
    ir_v2 = obs_compiler.compile(draft_v2)
    
    print(f"  Source Hash: {ir_v2.source_hash[:16]}...")
    print(f"  Sentences: {len(ir_v2.sentences)}")
    print(f"  Patterns: {len(ir_v2.patterns)}")

    # ---------- Step 3.7: Validator 再次检查 ----------
    print("\n[7] Validator → ComplianceReport v2")
    report_v2 = validator.validate(ir_v2, layer_targets)
    
    print(f"  Overall Compliance: {report_v2.overall_compliance:.2f}")
    for res in report_v2.layer_results:
        print(f"    {res.layer}: {'✅' if res.compliant else '❌'} (target={res.target_level})")

    # ---------- 结果对比 ----------
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)
    print(f"  Compliance v1: {report_v1.overall_compliance:.2f}")
    print(f"  Compliance v2: {report_v2.overall_compliance:.2f}")
    print(f"  改进: {'✅' if report_v2.overall_compliance > report_v1.overall_compliance else '❌'}")

    # 详细层对比
    for i, res_v1 in enumerate(report_v1.layer_results):
        res_v2 = report_v2.layer_results[i]
        print(f"    {res_v1.layer}: {res_v1.compliant} → {res_v2.compliant}")

    # ---------- 输出变化摘要 ----------
    print("\n--- Draft v1 → v2 变化摘要 ---")
    if draft_v1 != draft_v2:
        # 简单统计变化
        print(f"  文本长度变化: {len(draft_v2) - len(draft_v1)} 字符")
        # 检查是否插入了 "密信"
        if "密信" in draft_v2 and "密信" in draft_v1:
            count_v1 = draft_v1.count("密信")
            count_v2 = draft_v2.count("密信")
            print(f"  '密信' 出现次数: {count_v1} → {count_v2}")
        
        # 显示变化位置（简单 diff）
        print("\n  --- 变化的片段（前后 100 字符）---")
        # 找到第一个不同的位置
        min_len = min(len(draft_v1), len(draft_v2))
        diff_pos = 0
        for i in range(min_len):
            if draft_v1[i] != draft_v2[i]:
                diff_pos = i
                break
        if diff_pos > 0:
            start = max(0, diff_pos - 50)
            end = min(len(draft_v2), diff_pos + 100)
            print(f"  变化附近: ...{draft_v2[start:end]}...")
    else:
        print("  警告：Draft v1 和 v2 完全相同！")
        print("  这说明 LLM（或模拟器）没有执行任何修改。")

    return {
        "draft_v1": draft_v1,
        "draft_v2": draft_v2,
        "ir_v1": ir_v1,
        "ir_v2": ir_v2,
        "report_v1": report_v1,
        "report_v2": report_v2,
        "plan": plan,
        "rendered": rendered,
        "improved": report_v2.overall_compliance > report_v1.overall_compliance
    }


# ============================================================
# 4. 运行测试
# ============================================================

if __name__ == "__main__":
    result = run_end_to_end_test(DRAFT_V1, LAYER_TARGETS, use_mock_llm=True)
    
    # 保存结果到文件
    output_file = "reunion_low_end_to_end_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "compliance_v1": result["report_v1"].overall_compliance,
            "compliance_v2": result["report_v2"].overall_compliance,
            "improved": result["improved"],
            "draft_v1_length": len(result["draft_v1"]),
            "draft_v2_length": len(result["draft_v2"])
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")