"""LLM 统一接口。所有上游模块只依赖本接口，不感知后端差异。

支持三种后端（config.LLM_BACKEND 切换）：
    mock  —— 本地无 GPU 验证管线用（返回规则化结果）
    hf    —— transformers 加载开源模型（Qwen/Llama，服务器 GPU 上跑）
    vllm  —— vLLM 起的 OpenAI 兼容端点（服务器推荐）
"""
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, prompts, max_new_tokens: int | None = None,
                 temperature: float | None = None):
        """补全接口，支持单条与批量（保序）。

        prompts: str | list[str]
        返回:   str（输入单条时）| list[str]（输入列表时）
        """
        raise NotImplementedError
