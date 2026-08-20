"""vLLM 后端：连服务器上 vLLM 起的 OpenAI 兼容端点（http://host:8000/v1）。

服务器侧启动见 scripts/start_vllm.sh。
客户端只需 openai 库 + base_url / api_key（config 或环境变量注入）。
batch 推理：线程池并发发送（服务端 continuous batching 自动聚合），保序返回。
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from config import LLM_BATCH_SIZE, VLLM_API_KEY, VLLM_URL
from llm.base import LLMBackend


class VLLMBackend(LLMBackend):
    name = "vllm"

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 max_workers: int | None = None):
        self.client = OpenAI(
            base_url=base_url or VLLM_URL,
            api_key=api_key or VLLM_API_KEY,
            timeout=300,
            max_retries=2,
        )
        self.max_workers = max_workers or LLM_BATCH_SIZE
        print(f"[VLLMBackend] endpoint {base_url or VLLM_URL} (workers={self.max_workers})")

    def _call(self, p: str, max_new: int, temp: float) -> str:
        resp = self.client.chat.completions.create(
            model="default",  # vLLM 端点不校验模型名（--served-model-name default）
            messages=[{"role": "user", "content": p}],
            max_tokens=max_new,
            temperature=temp,
        )
        return resp.choices[0].message.content.strip()

    def complete(self, prompts, max_new_tokens: int | None = None,
                 temperature: float | None = None):
        from config import LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE
        single = isinstance(prompts, str)
        texts = [prompts] if single else list(prompts)

        max_new = max_new_tokens or LLM_MAX_NEW_TOKENS
        temp = LLM_TEMPERATURE if temperature is None else temperature

        outputs = [None] * len(texts)
        if len(texts) == 1:
            outputs[0] = self._call(texts[0], max_new, temp)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures = {ex.submit(self._call, p, max_new, temp): i
                           for i, p in enumerate(texts)}
                for f in as_completed(futures):
                    outputs[futures[f]] = f.result()
        return outputs[0] if single else outputs
