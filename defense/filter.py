"""风险约束上下文选择（P3）：高风险联盟 → 剔除对应文档 → 安全上下文。

策略：
    1) 每个联盟算风险分（RuleRisk / LearnedRisk）
    2) risk >= tau 的联盟 → 其 claim 所在文档全部剔除
    3) 剩余文档作为安全上下文；全部剔除 → 拒答（refuse=True）

联盟 → 文档映射：claim.meta["doc_id"]（评估管线构造 claim 时记录）。
"""
from __future__ import annotations

from defense.risk import feature_matrix


def select_context(docs: list[dict], claims: list[dict], coalitions: list[dict],
                   risk_fn, tau: float = 0.5) -> dict:
    """返回 {kept_docs, removed_docs, refused, risks}。

    docs:      检索返回的文档列表 [{doc_id, text, meta}]
    claims:    [{claim_id, text, meta:{doc_id, poisoned}}]
    coalitions:[{claims:[claim_id...], 特征...}]（feature_matrix 可读）
    """
    feats = feature_matrix(coalitions)
    risks = risk_fn.predict_proba(feats).tolist() if len(coalitions) else []

    cid2doc = {c["claim_id"]: c["meta"].get("doc_id") for c in claims}
    drop_claims = set()
    for c, r in zip(coalitions, risks):
        if r >= tau:
            drop_claims.update(c["claims"])

    drop_doc_ids = {cid2doc.get(cid) for cid in drop_claims}
    drop_doc_ids.discard(None)
    kept = [d for d in docs if d["doc_id"] not in drop_doc_ids]
    removed = [d for d in docs if d["doc_id"] in drop_doc_ids]
    return {
        "kept_docs": kept,
        "removed_docs": removed,
        "refused": len(kept) == 0,
        "risks": [round(r, 4) for r in risks],
    }


__all__ = ["select_context"]
