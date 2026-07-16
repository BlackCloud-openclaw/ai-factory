"""
PatchRenderer: IR 到自然语言边界层
Phase 6.3C Step 4 — 支持 REPLACE_SENTENCE
修正：Single Output Contract Principle
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from src.runtime.edit_compiler import EditPlan, EditAction, EditOperation
from src.runtime.observation_compiler import ObservationIR, SentenceSpan, PatternSpan


# ============================================================
# 1. Patch Prompt 数据结构（Renderer 输出）
# ============================================================

@dataclass
class RenderedPatch:
    """Renderer 的输出：结构化编辑指令"""
    system_prompt: str
    edit_instructions: List[str]
    preserve_constraints: List[str]
    full_prompt: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "edit_instructions": self.edit_instructions,
            "preserve_constraints": self.preserve_constraints,
            "full_prompt": self.full_prompt
        }


# ============================================================
# 2. PatchRenderer
# ============================================================

class PatchRenderer:
    """
    PatchRenderer 是自然语言边界的最后一道关卡。
    职责：将 ID 引用的 EditPlan 转换为 LLM 可执行的精确指令。
    
    修正 (Phase 6 收束)：
    - 遵循 Single Output Contract Principle
    - 所有 Prompt 尾部统一要求 JSON 格式输出
    - 与 Grammar 约束对齐（JSON），消除输出契约分歧
    """

    def __init__(self):
        self.system_prompt_template = """你是一个专业的文本编辑助手。你的任务是根据以下编辑指令，精确修改给定的文本。

**核心原则**：
1. 只执行指定的编辑操作，不要做额外的润色或改写。
2. 保持原文中未提及部分完全不变。
3. 编辑时保持原文的叙事风格和语气一致性。
4. 输出时，请严格按照要求的格式输出。"""

        self.instruction_templates = {
            EditOperation.INSERT_AFTER: (
                "在以下句子之后插入一段新内容（包含关键词「{keyword}」）：\n"
                "锚点句子：\n"
                "「{anchor_text}」\n"
                "插入要求：\n"
                "- 新内容必须包含关键词「{keyword}」。\n"
                "- 新内容应与锚点句子的逻辑连贯。\n"
                "- 如果锚点句子包含类似「因为/所以/这意味着」等连接词，新内容应承接其推理方向。"
            ),
            EditOperation.INSERT_BEFORE: (
                "在以下句子之前插入一段新内容（包含关键词「{keyword}」）：\n"
                "锚点句子：\n"
                "「{anchor_text}」\n"
                "插入要求：\n"
                "- 新内容必须包含关键词「{keyword}」。\n"
                "- 新内容应与锚点句子的逻辑连贯。"
            ),
            EditOperation.REPLACE_SENTENCE: (
                "将以下句子替换为一个新的完整句子：\n"
                "原句子：\n"
                "「{anchor_text}」\n"
                "替换要求：\n"
                "- 新句子必须同时包含关键词「{keyword1}」和「{keyword2}」。\n"
                "- 新句子应保留原句子的基本逻辑流向和叙事角色。\n"
                "- 新句子应是一个完整、自然的中文句子，句号结尾。\n"
                "- 新句子中，{keyword1} 和 {keyword2} 应存在因果或推理关系。"
            ),
        }

    def render(self, edit_plan: EditPlan, observation_ir: ObservationIR) -> RenderedPatch:
        sentence_map = {s.id: s for s in observation_ir.sentences}
        pattern_map = {p.id: p for p in observation_ir.patterns}

        edit_instructions = []
        preserve_constraints = []

        for idx, action in enumerate(edit_plan.actions):
            anchor_sentence = sentence_map.get(action.anchor_sentence_id)
            if not anchor_sentence:
                continue

            anchor_text = anchor_sentence.text
            template = self.instruction_templates.get(action.operation)
            if not template:
                template = "执行编辑操作：在锚点句子 {anchor_text} 处进行修改。"

            if action.payload_type == "combined":
                keyword1, keyword2 = self._get_combined_keywords(observation_ir, action.anchor_sentence_id)
                instruction = template.format(
                    anchor_text=anchor_text,
                    keyword1=keyword1,
                    keyword2=keyword2
                )
            else:
                keyword = self._get_keyword_example(
                    payload_type=action.payload_type,
                    observation_ir=observation_ir,
                    anchor_sentence_id=action.anchor_sentence_id
                )
                instruction = template.format(
                    anchor_text=anchor_text,
                    keyword=keyword
                )

            edit_instructions.append(f"[编辑 {idx+1}] {instruction}")

            # 保留约束
            for sent_id in action.preserve_sentence_ids:
                if sent_id != action.anchor_sentence_id:
                    sent = sentence_map.get(sent_id)
                    if sent:
                        preserve_constraints.append(f"- 保留句子：「{sent.text[:30]}...」")

            for pattern_id in action.preserve_pattern_ids:
                pat = pattern_map.get(pattern_id)
                if pat:
                    preserve_constraints.append(f"- 保留关键词/特征：「{pat.text}」")

        preserve_constraints = list(set(preserve_constraints))

        system_prompt = self.system_prompt_template + "\n\n" + "\n".join(preserve_constraints) if preserve_constraints else self.system_prompt_template
        instructions_section = "\n\n".join(edit_instructions) if edit_instructions else "本次不需要编辑。"

        # ============================================================
        # Phase 6 收束：Single Output Contract Principle
        # 统一输出契约：所有 Patch 指令都要求 JSON 格式
        # 与 Grammar 约束对齐，消除输出契约分歧
        # ============================================================
        full_prompt = f"""{system_prompt}

---
【编辑任务】：
{instructions_section}

---
【输出要求】：
请严格按照以下 JSON 格式输出修改后的结果：
{{
    "revised_text": "修改后的完整文本（包含所有修改内容）"
}}
不要添加任何额外解释、注释或前缀。直接输出 JSON。
"""

        return RenderedPatch(
            system_prompt=system_prompt,
            edit_instructions=edit_instructions,
            preserve_constraints=preserve_constraints,
            full_prompt=full_prompt
        )

    # ---------- 辅助方法 ----------
    def _get_keyword_example(self, payload_type: str, observation_ir: ObservationIR, anchor_sentence_id: str) -> str:
        for p in observation_ir.patterns:
            if p.sentence_id == anchor_sentence_id and p.pattern_type == payload_type:
                return p.text
        for p in observation_ir.patterns:
            if p.pattern_type == payload_type:
                return p.text
        return f"[{payload_type} 关键词]"

    def _get_combined_keywords(self, observation_ir: ObservationIR, anchor_sentence_id: str) -> tuple:
        anchor_patterns = [p for p in observation_ir.patterns if p.sentence_id == anchor_sentence_id]
        state_example = None
        logic_example = None

        for p in anchor_patterns:
            if p.pattern_type == "state_keyword" and not state_example:
                state_example = p.text
            if p.pattern_type == "logic_marker" and not logic_example:
                logic_example = p.text

        if not state_example:
            for p in observation_ir.patterns:
                if p.pattern_type == "state_keyword":
                    state_example = p.text
                    break
        if not logic_example:
            for p in observation_ir.patterns:
                if p.pattern_type == "logic_marker":
                    logic_example = p.text
                    break

        if not state_example:
            state_example = "密信"
        if not logic_example:
            logic_example = "因此"

        return (state_example, logic_example)


# ============================================================
# 3. 快速测试
# ============================================================

if __name__ == "__main__":
    from src.runtime.observation_compiler import ObservationCompiler
    from src.runtime.validator import Validator
    from src.runtime.edit_compiler import EditCompiler

    sample_draft = """林逸的指节在袖中捏紧又松开三次。那封泛黄的密信在怀中发烫，墨迹晕染处仍能辨认出"天机阁地底第七重"的批注——正是师兄失踪前夜用特殊药水写下的。此刻那道背影肩宽腰窄，与十年前领他入门时身形分毫不差，连负手而立的习惯都未曾改变。

"你留下的信里说'血色月华'是假象。"林逸听见自己嗓音发涩，"可那天值夜的弟子明明看见红月亮贯穿三更天。"风卷起几片零落桃花，落在那人玄色衣摆上。他忽然想起师兄被除名那日，宗门大殿外也是这样漫天花雨。"""

    obs_compiler = ObservationCompiler()
    ir = obs_compiler.compile(sample_draft)

    validator = Validator()
    layer_targets = {
        "reasoning": "enhanced",
        "justification": "enhanced",
        "construction": "enhanced"
    }
    report = validator.validate(ir, layer_targets)

    edit_compiler = EditCompiler()
    plan = edit_compiler.compile(ir, report, diagnosis_id="D001")

    renderer = PatchRenderer()
    rendered = renderer.render(plan, ir)

    print("=" * 60)
    print("Rendered Patch (Prompt 模板)")
    print("=" * 60)
    print(f"Edit Instructions ({len(rendered.edit_instructions)}):")
    for i, instr in enumerate(rendered.edit_instructions):
        print(f"{i+1}. {instr[:200]}...")
    print("\n--- Full Prompt (末尾部分) ---")
    print(rendered.full_prompt[-500:])