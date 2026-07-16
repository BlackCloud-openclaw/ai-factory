# src/runtime/renderers/prompt_renderer.py
"""
Prompt Renderer - LayerControlTargets → Prompt

职责：将 Runtime IR (LayerControlTargets) 渲染为 Writer 可执行的 Prompt。
渲染器不包含业务逻辑，只负责文本生成。
"""

from dataclasses import dataclass
from typing import Dict, Any

from src.runtime.compiler import LayerControlTargets, LayerTarget
from src.runtime.models import SceneAnalysis, PolicyConfig


@dataclass
class RenderContext:
    """渲染上下文 - 包含生成 Prompt 所需的所有信息"""
    layer_targets: LayerControlTargets
    state: str
    scene_analysis: SceneAnalysis
    policy: PolicyConfig
    scene_a: str
    description: str = "请从以上场景继续，写一段林逸做出决定的场景。"


class PromptRenderer:
    """
    Prompt Renderer v1.0
    
    将 LayerControlTargets 渲染为 Writer Prompt。
    不包含任何逻辑判断，只负责文本模板。
    """
    
    VERSION = "1.0"
    TEMPLATE = "layered_control_v1"
    
    def render(self, context: RenderContext) -> str:
        """渲染 Prompt"""
        
        # 1. 生成各层指令
        l1_instruction = self._render_prediction(context.layer_targets.prediction)
        l2_instruction = self._render_reasoning(context.layer_targets.reasoning)
        l3_instruction = self._render_justification(context.layer_targets.justification)
        l4_instruction = self._render_construction(context.layer_targets.construction)
        
        # 2. 组装
        return f"""你是一位小说作者。请根据以下场景开头，续写一段场景正文（300-500字）。

【场景开头】
{context.scene_a}

【状态信息】
{context.state}

【分层控制目标 - 必须遵守】

L1 - Prediction（事件选择）：
{l1_instruction}

L2 - Reasoning（角色推理）：
{l2_instruction}

L3 - Justification（决策理由）：
{l3_instruction}

L4 - Construction（叙事实现）：
{l4_instruction}

【续写要求】
{context.description}

写作要求：
1. 保持与开头一致的第三人称叙述风格
2. 必须体现角色形成决策的思考过程
3. 结尾必须出现明确的行动倾向
4. 不要添加任何解释或元评论
5. 只输出续写的正文，不要包含任何额外内容"""
    
    def _render_prediction(self, target: LayerTarget) -> str:
        if target == LayerTarget.FIXED:
            return "State 不得改变已选定的事件。事件走向已经锁定。"
        elif target == LayerTarget.ASSIST:
            return "State 可辅助事件选择，但不强制改变。"
        elif target == LayerTarget.PRIMARY:
            return "State 应作为决定事件选择的主要依据。"
        elif target == LayerTarget.NORMAL:
            return "State 可正常参与事件选择。"
        elif target == LayerTarget.NONE:
            return "State 不涉及事件选择。"
        else:
            return "State 可正常参与事件选择。"
    
    def _render_reasoning(self, target: LayerTarget) -> str:
        if target == LayerTarget.ENHANCED:
            return "State 必须出现在角色的推理过程中，解释为什么选择这个事件是合理的。"
        elif target == LayerTarget.NORMAL:
            return "State 可在角色的推理中被提及。"
        elif target == LayerTarget.NONE:
            return "State 不涉及角色推理。"
        else:
            return "State 可在角色的推理中被提及。"
    
    def _render_justification(self, target: LayerTarget) -> str:
        if target == LayerTarget.ENHANCED:
            return "State 必须成为决策理由的一部分。"
        elif target == LayerTarget.NORMAL:
            return "State 可成为决策理由的一部分。"
        elif target == LayerTarget.NONE:
            return "State 不涉及决策理由。"
        else:
            return "State 可成为决策理由的一部分。"
    
    def _render_construction(self, target: LayerTarget) -> str:
        if target == LayerTarget.ENHANCED:
            return "State 必须影响叙事实现的方式。"
        elif target == LayerTarget.NORMAL:
            return "State 可影响叙事实现的方式。"
        elif target == LayerTarget.NONE:
            return "State 不涉及叙事实现。"
        else:
            return "State 可影响叙事实现的方式。"


def render_prompt(context: RenderContext) -> str:
    """便捷函数"""
    return PromptRenderer().render(context)