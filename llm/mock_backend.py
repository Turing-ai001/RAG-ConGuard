"""Mock 后端：本地无 GPU / 无服务器时验证管线用。

响应规则（按优先级）：
    1. handler 函数（smoke test 自定义，如返回合法 claim JSON）
    2. substring_map：prompt 包含某子串 → 固定响应
    3. 默认规则：提取 prompt 首句作为回答
所有规则确定性输出，保证测试可复现。
"""
from typing import Callable, Dict, Optional

from llm.base import LLMBackend


class MockBackend(LLMBackend):
    name = "mock"

    def __init__(self, handler: Optional[Callable[[str], str]] = None,
                 substring_map: Optional[Dict[str, str]] = None):
        self.handler = handler
        self.substring_map = substring_map or {}

    def _respond(self, prompt: str) -> str:
        if self.handler is not None:
            return self.handler(prompt)
        for key, resp in self.substring_map.items():
            if key in prompt:
                return resp
        first_sentence = prompt.strip().split("\n")[0][:200]
        return f"[MOCK] {first_sentence}"

    def complete(self, prompts, max_new_tokens: int | None = None,
                 temperature: float | None = None):
        single = isinstance(prompts, str)
        texts = [prompts] if single else list(prompts)
        outputs = [self._respond(p) for p in texts]
        return outputs[0] if single else outputs
