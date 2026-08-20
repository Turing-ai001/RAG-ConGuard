"""HuggingFace 本地模型后端（transformers）。

懒加载：首次调用才加载模型，mock/vllm 环境下不占内存。
batch 推理：右 padding + batch generate，按 LLM_BATCH_SIZE 分块。
离线加载：HF_OFFLINE=1 或 local_files_only=True（模型已缓存到本地时，避免联网检查超时）。
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, DTYPE, HF_MODEL, LLM_BATCH_SIZE
from llm.base import LLMBackend


class HFBackend(LLMBackend):
    name = "hf"

    def __init__(self, model_name: str | None = None,
                 local_files_only: bool | None = None):
        self.model_name = model_name or HF_MODEL
        offline = os.environ.get("HF_OFFLINE", "").lower() in ("1", "true", "yes")
        self.local_files_only = (offline if local_files_only is None
                                 else local_files_only)
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        print(f"[HFBackend] loading {self.model_name} on {DEVICE} "
              f"(local_files_only={self.local_files_only}) ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, local_files_only=self.local_files_only)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # decoder-only 模型 batch 推理必须左 padding：右 padding 会让模型
        # 把 pad token 当起始 token，导致生成退化（回显 prompt / 乱码）
        self._tokenizer.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=DTYPE,
            device_map="auto" if DEVICE == "cuda" else None,
            local_files_only=self.local_files_only,
        )
        if DEVICE == "cuda" and not hasattr(self._model, "hf_device_map"):
            self._model = self._model.to(DEVICE)
        self._model.eval()
        print("[HFBackend] ready.")

    def _format(self, prompt: str) -> str:
        """有 chat template 则走模板，否则原样输入。"""
        if self._tokenizer.chat_template:
            try:
                return self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass
        return prompt

    def complete(self, prompts, max_new_tokens: int | None = None,
                 temperature: float | None = None):
        from config import LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE
        self._ensure_loaded()
        single = isinstance(prompts, str)
        texts = [prompts] if single else list(prompts)

        max_new = max_new_tokens or LLM_MAX_NEW_TOKENS
        temp = LLM_TEMPERATURE if temperature is None else temperature

        outputs = []
        for i in range(0, len(texts), LLM_BATCH_SIZE):
            chunk = [self._format(p) for p in texts[i:i + LLM_BATCH_SIZE]]
            inputs = self._tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=4096,
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.no_grad():
                gen = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    temperature=temp,
                    do_sample=temp > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            for j, seq in enumerate(gen):
                out = self._tokenizer.decode(
                    seq[inputs["input_ids"][j].shape[0]:], skip_special_tokens=True
                )
                outputs.append(out.strip())
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        return outputs[0] if single else outputs
