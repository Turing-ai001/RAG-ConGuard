"""LLM-as-judge 关系判定（方案 B 兜底）：NLI 低置信的 claim 对交给 LLM 重判。

LLM 输出 JSON：{"relation": "support|refute|entail|contradict|unrelated", "confidence": 0.8}
解析三层容错（与 claim/extractor.py 同款思路）。
"""
from __future__ import annotations

import json
import re
from typing import Optional

from config import LLM_MAX_NEW_TOKENS
from llm.base import LLMBackend
from relation.types import Relation, RelationType

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

RELATION_PROMPT = (
    "You are a relation judge for factual claims. Given two claims extracted "
    "from documents, decide the relation between Claim A (premise) and Claim B (hypothesis).\n"
    "Options:\n"
    '- "support": B supports A (consistent, B reinforces A)\n'
    '- "refute": B refutes A (B contradicts or undermines A)\n'
    '- "entail": B is logically entailed by A (A implies B)\n'
    '- "contradict": A and B express conflicting facts (cannot both be true)\n'
    '- "unrelated": neither supports nor contradicts (different topics)\n'
    'Return ONLY a JSON object: {"relation": "...", "confidence": 0.0-1.0}\n\n'
    "Claim A: {src}\n\nClaim B: {dst}"
)

_VALID = {t.value for t in RelationType}


class LLMJudge:
    """LLM 兜底判定器。judge_edges 返回 Relation 列表。"""

    def __init__(self, llm: LLMBackend, prompt_template: Optional[str] = None):
        self.llm = llm
        self.prompt_template = prompt_template or RELATION_PROMPT

    # ---------- 主接口 ----------
    def judge_edges(self, src_texts: list[str], dst_texts: list[str],
                    src_ids: list[str], dst_ids: list[str],
                    batch_size: int | None = None) -> list[Relation]:
        """批量判定 (src → dst) 关系。"""
        from config import LLM_BATCH_SIZE
        bs = batch_size or LLM_BATCH_SIZE
        edges: list[Relation] = []
        for i in range(0, len(src_texts), bs):
            chunk_s = src_texts[i:i + bs]
            chunk_d = dst_texts[i:i + bs]
            prompts = [self._fill(self.prompt_template, s, d)
                       for s, d in zip(chunk_s, chunk_d)]
            responses = self.llm.complete(prompts, max_new_tokens=LLM_MAX_NEW_TOKENS)
            if isinstance(responses, str):
                responses = [responses]
            for (s_id, d_id), (s, d), resp in zip(
                    zip(src_ids[i:i + bs], dst_ids[i:i + bs]), zip(chunk_s, chunk_d), responses):
                rel, conf = self._parse(resp)
                edges.append(Relation(s_id, d_id, rel, conf, provenance="llm"))
        return edges

    @staticmethod
    def _fill(template: str, src: str, dst: str) -> str:
        return (template.replace("{src}", src).replace("{dst}", dst))

    # ---------- 解析（三层容错） ----------
    def _parse(self, resp: str) -> tuple[RelationType, float]:
        rel = RelationType.UNRELATED
        conf = 0.5
        raw = self._extract_json(resp)
        if raw is not None:
            try:
                obj = json.loads(raw)
                r = str(obj.get("relation", "")).lower().strip()
                if r in _VALID:
                    rel = RelationType(r)
                conf = min(1.0, max(0.0, float(obj.get("confidence", 0.5))))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        else:
            # 兜底：扫描响应文本里的关系词
            lower = resp.lower()
            for t in (RelationType.ENTAIL, RelationType.CONTRADICT,
                      RelationType.SUPPORT, RelationType.REFUTE):
                if t.value in lower:
                    rel = t
                    break
        return rel, conf

    @staticmethod
    def _extract_json(resp: str) -> Optional[str]:
        m = _FENCE.search(resp)
        if m:
            return m.group(1).strip()
        start, end = resp.find("{"), resp.rfind("}")
        if start != -1 and end > start:
            return resp[start:end + 1]
        return None
