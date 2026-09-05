# -*- coding: utf-8 -*-
"""fp/selectors.py — 安全上下文选择策略（冻结协议 §6，Phase 2 重点）。

统一接口：select(docs, claims, coalitions, risk_fn, param) -> dict
    {kept_docs, removed_docs, refused, risks, removed_coalitions}

S0  whole-coalition threshold removal（= 旧 filter.select_context 行为）
S1  risk-ranked removal：只删风险最高联盟（其余证据保留）
S2  greedy safe context reconstruction：按风险降序删除联盟，但保留至少
    lambda * len(docs) 的文档（λ∈(0,1] 在 val 上调）；文档保留不足下限时
    拒绝溢出删除（宁可留少量高风险证据，也不要空上下文）——utility 保护。
"""
from __future__ import annotations

from defense.risk import feature_matrix


def _risks(coalitions: list[dict], risk_fn) -> list[float]:
    feats = feature_matrix(coalitions)
    return risk_fn.predict_proba(feats).tolist() if len(coalitions) else []


def is_coalition(rec: dict) -> bool:
    """M5 精化（v1.2）：最终联盟 = 候选且满足 Score>0 口径
    （Score = α·Density + β·I − γ·IndepSupp，默认 α=β=γ=1，等价条件：
    size≥2 且 (cohesion>0 或 impact>0)——孤立单点无内部依赖又无影响，非联盟）。"""
    return rec.get("size", 1) >= 2 and (
        rec.get("cohesion", 0.0) > 0 or rec.get("impact", 0.0) > 0)


def _apply(docs, claims, coalitions, risk_fn, pick):
    """pick(risks) -> 被删联盟下标集合（仅对"最终联盟"实施删除，单点/非联盟豁免）。"""
    cid2doc = {c["claim_id"]: c["meta"].get("doc_id") for c in claims}
    coalitions = [c for c in coalitions if is_coalition(c)]
    risks = _risks(coalitions, risk_fn)
    drop = pick(risks)
    drop_claims = set()
    for i in drop:
        drop_claims.update(coalitions[i]["claims"])
    drop_doc_ids = {cid2doc.get(cid) for cid in drop_claims}
    drop_doc_ids.discard(None)
    kept = [d for d in docs if d["doc_id"] not in drop_doc_ids]
    removed = [d for d in docs if d["doc_id"] in drop_doc_ids]
    return {
        "kept_docs": kept,
        "removed_docs": removed,
        "refused": len(kept) == 0,
        "risks": [round(r, 4) for r in risks],
        "removed_coalitions": [
            {"claims": coalitions[i]["claims"], "risk": round(risks[i], 4)}
            for i in drop
        ],
    }


def select_s0(docs, claims, coalitions, risk_fn, tau: float = 0.5) -> dict:
    def pick(risks):
        return [i for i, r in enumerate(risks) if r >= tau]
    return _apply(docs, claims, coalitions, risk_fn, pick)


def select_s1(docs, claims, coalitions, risk_fn, tau: float = 0.5) -> dict:
    def pick(risks):
        if not risks or max(risks) < tau:
            return []
        return [int(max(range(len(risks)), key=lambda i: risks[i]))]
    return _apply(docs, claims, coalitions, risk_fn, pick)


def select_s2(docs, claims, coalitions, risk_fn, lam: float = 0.6,
              tau: float = 0.0) -> dict:
    """贪婪重建：按风险降序尝试删除，仅当保留文档数 ≥ lam*|docs| 才执行。"""
    n_docs = len(docs)
    floor = max(1, int(lam * n_docs)) if lam < 1.0 else n_docs

    def pick(risks):
        order = sorted(range(len(risks)), key=lambda i: -risks[i])
        drop, removed_cnt = [], 0
        for i in order:
            risks_i = risks[i]
            if risks_i < tau:
                break                    # 剩余联盟风险低到不值得删
            csize = len(coalitions[i]["claims"])
            if n_docs - removed_cnt - csize >= floor:
                drop.append(i)
                removed_cnt += csize
            else:
                break                    # 会跌破保留下限 → 停
        return drop
    return _apply(docs, claims, coalitions, risk_fn, pick)


SELECTORS = {"s0": select_s0, "s1": select_s1, "s2": select_s2}
