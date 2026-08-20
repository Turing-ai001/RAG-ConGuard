"""LLM 统一封装：get_backend() 工厂，按 config.LLM_BACKEND 返回对应后端。"""
from config import LLM_BACKEND
from llm.base import LLMBackend
from llm.hf_backend import HFBackend
from llm.mock_backend import MockBackend
from llm.vllm_backend import VLLMBackend


def get_backend(name: str | None = None) -> LLMBackend:
    """按名字（默认 config.LLM_BACKEND）创建后端实例。"""
    n = (name or LLM_BACKEND).lower()
    if n == "mock":
        return MockBackend()
    if n == "hf":
        return HFBackend()
    if n == "vllm":
        return VLLMBackend()
    raise ValueError(f"unknown LLM backend: {n} (mock|hf|vllm)")


__all__ = ["LLMBackend", "get_backend", "HFBackend", "MockBackend", "VLLMBackend"]
