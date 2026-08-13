# src/narrative/realizers/interfaces.py

from typing import Protocol, runtime_checkable


@runtime_checkable
class TextGenerator(Protocol):
    """
    文本生成器接口 — ReferenceRealizer 的唯一依赖

    Realizer 不需要知道 OpenAI、vLLM、Anthropic 的区别。
    它只需要一个「输入 Prompt → 输出文本」的能力。
    """
    async def generate(self, prompt: str) -> str:
        """
        根据 Prompt 生成文本

        Args:
            prompt: 输入提示

        Returns:
            str: 生成的文本
        """
        ...