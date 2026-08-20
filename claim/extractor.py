"""主张提取模块（P0：prompt engineering 版）。

输入：query + 检索出的文档列表
输出：每个文档的与 query 相关的 claims，结构化为 JSON。
解析做了三层容错：
    1. 直接 json.loads
    2. 剥掉代码围栏 / 取首个 { } 块
    3. 整段按句切分兜底（标注 relevance 0.5）
"""
import json
import re
from typing import Optional

from config import CLAIM_MAX_CLAIMS, CLAIM_PROMPT, CLAIM_RELEVANCE_THRESHOLD
from llm.base import LLMBackend

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


class ClaimExtractor:
    def __init__(self, llm: LLMBackend, prompt_template: Optional[str] = None,
                 max_claims: int = CLAIM_MAX_CLAIMS,
                 threshold: float = CLAIM_RELEVANCE_THRESHOLD):
        self.llm = llm
        self.prompt_template = prompt_template or CLAIM_PROMPT
        self.max_claims = max_claims
        self.threshold = threshold

    # ---------- 主接口 ----------
    @staticmethod
    def _fill(template: str, query: str, doc: str) -> str:
        """填充模板。用 replace 而非 str.format —— 模板内含 JSON 字面花括号
        （如 {"claims": ...}），format 会将其误当占位符（KeyError 坑）。"""
        return (template.replace("{query}", query)
                        .replace("{doc}", doc))

    def extract(self, query: str, docs: list[dict]) -> list[dict]:
        """docs: [{doc_id, text, ...}]，返回 [{doc_id, claims: [{text, relevance}]}]"""
        prompts = [self._fill(self.prompt_template, query, d["text"]) for d in docs]
        responses = self.llm.complete(prompts)
        if isinstance(responses, str):  # 单文档兜底
            responses = [responses]
        results = []
        for doc, resp in zip(docs, responses):
            claims = self._parse(resp)
            claims = [c for c in claims if c["relevance"] >= self.threshold][:self.max_claims]
            results.append({"doc_id": doc["doc_id"], "claims": claims})
        return results

    def extract_one(self, query: str, doc_text: str) -> list[dict]:
        """单文档版本，返回 claims 列表。"""
        resp = self.llm.complete(self._fill(self.prompt_template, query, doc_text))
        claims = self._parse(resp)
        return [c for c in claims if c["relevance"] >= self.threshold][:self.max_claims]

    # ---------- 解析（三层容错） ----------
    def _parse(self, resp: str) -> list[dict]:
        raw = self._extract_json(resp)
        if raw is not None:
            try:
                obj = json.loads(raw)
                claims = obj.get("claims", obj if isinstance(obj, list) else [])
                return [
                    {"text": str(c.get("text", "")).strip(),
                     "relevance": float(c.get("relevance", 0.5))}
                    for c in claims if isinstance(c, dict) and str(c.get("text", "")).strip()
                ]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        # 兜底：按句切分
        return [{"text": s.strip(), "relevance": 0.5}
                for s in _SENT_SPLIT.split(resp.strip()) if s.strip()]

    @staticmethod
    def _extract_json(resp: str) -> Optional[str]:
        m = _FENCE.search(resp)
        if m:
            return m.group(1).strip()
        start, end = resp.find("{"), resp.rfind("}")
        if start != -1 and end > start:
            return resp[start:end + 1]
        return None
