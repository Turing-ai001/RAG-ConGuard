"""端到端防御管线（P3）：query → 检索 → 图 → 联盟 → 风险 → 安全生成。

claim 口径与 P2 评估一致（检索文档文本截断 220 作为 claim），保证实验可比；
claim.meta 记录 doc_id 供 filter 映射剔除。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from config import RAG_PROMPT, TOP_K
from defense.filter import select_context
from graph import ClaimGraph
from relation import RelationExtractor


def claims_from_docs(docs: list[dict], poisoned_suffix: str = ":poisoned:") -> list[dict]:
    """检索文档 → claim 列表（meta 记录 doc_id / poisoned）。"""
    out = []
    for d in docs:
        is_poisoned = poisoned_suffix in d["doc_id"]
        out.append({
            "claim_id": f"{d['doc_id'].split(':')[-2][:8]}-{len(out)}",
            "text": d["text"][:220],
            "meta": {"doc_id": d["doc_id"], "poisoned": is_poisoned,
                     "target_query_id": d.get("meta", {}).get("target_query_id", "")},
        })
    return out


class DefensePipeline:
    """有防御 / 无防御双路径。风险学习器与 τ 由外部注入。"""

    def __init__(self, kb, embedder, extractor: RelationExtractor,
                 llm, risk_fn, tau: float = 0.5):
        self.kb = kb
        self.embedder = embedder
        self.extractor = extractor
        self.llm = llm
        self.risk_fn = risk_fn
        self.tau = tau

    @staticmethod
    def _context(docs) -> str:
        return "\n".join(d["text"] for d in docs)

    def run_baseline(self, query: str, top_k: int = TOP_K) -> dict:
        """无防御：直接生成（对照基线）。"""
        docs = self.kb.search(query, top_k=top_k)
        answer = self.llm.complete(
            RAG_PROMPT.replace("[question]", query)
                      .replace("[context]", self._context(docs)))
        return {"answer": answer, "docs": docs}

    def run_defended(self, query: str, top_k: int = TOP_K,
                     answer_sims: Optional[np.ndarray] = None,
                     answer_ids: Optional[list[str]] = None) -> dict:
        """防御：检索 → claim → 图 → 联盟 → 特征 → 风险 → 剔除/拒答 → 生成。

        answer_sims/answer_ids：候选答案 BGE 相似度矩阵（(n_claims, n_ans)）及其 id；
        传入时计算反事实影响/控制力，缺失时 impact=0（仅结构特征）。
        """
        docs = self.kb.search(query, top_k=top_k)
        claims = claims_from_docs(docs)
        edges = self.extractor.extract_all(claims)
        g = ClaimGraph(claims, edges)

        from coalition.analyze import coalition_features
        from coalition.trust import counterfactual_impacts
        coalitions = g.coalition_nodes("signed")
        feats = coalition_features(g, coalitions)
        fmap = {tuple(sorted(f["claims"])): f for f in feats}

        impacts = (counterfactual_impacts(g, coalitions, claims, answer_sims,
                                          answer_ids or ["x", "y"])
                   if answer_sims is not None else [])
        # counterfactual_impacts 过滤 size<min 的联盟 → 用字典对齐，防止 zip 错位
        impact_map = {tuple(imp["claims"]): imp for imp in impacts}

        records = []
        for comp in coalitions:
            imp = impact_map.get(tuple(sorted(comp)),
                                 {"impact": 0.0, "claims": sorted(comp)})
            feat = fmap.get(tuple(sorted(comp)), {})
            corr = feat.get("external_corroboration", 0.0)
            control = imp["impact"] * (1 - min(corr, 1.0)) if corr else imp["impact"]
            records.append({
                "claims": sorted(comp),
                "size": len(comp),
                "cohesion": feat.get("cohesion", 0.0),
                "external_corroboration": feat.get("external_corroboration", 0.0),
                "external_refute": feat.get("external_refute", 0.0),
                "impact": float(imp["impact"]),
                "control": round(float(control), 4),
            })

        sel = select_context(docs, claims, records, self.risk_fn, self.tau)
        if sel["refused"]:
            return {"answer": "I don't know", "docs": sel["kept_docs"],
                    "removed_docs": sel["removed_docs"], "refused": True,
                    "risks": sel["risks"]}
        answer = self.llm.complete(
            RAG_PROMPT.replace("[question]", query)
                      .replace("[context]", self._context(sel["kept_docs"])))
        return {"answer": answer, "docs": sel["kept_docs"],
                "removed_docs": sel["removed_docs"], "refused": False,
                "risks": sel["risks"]}
