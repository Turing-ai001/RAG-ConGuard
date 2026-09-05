# -*- coding: utf-8 -*-
"""fp/baselines_def.py — B4 NLI-consistency / B6 per-doc LOO 反事实基线（Phase 4）。

实现口径（公平、公开定义、差异记录在结果 config）：
  B4 ReliabilityRAG 风格（NLI 一致性）：
     矛盾图（NLI contradiction 边，置信 ≥0.5）+ 一致多数选择。
     实现：找出全部 contradiction 关系后，取"正边连通分量中最大支持簇"；
     若不存在矛盾边 → 保留全部文档（保底）；有矛盾 → 保留最大支持簇文档。
     与 ReliabilityRAG 的差异：其用 graph-based ranking + 子集选择，
     我们以公开可见假设的最小充分实现（一致簇 + 最大簇）复现——记录差异。

  B6 Per-document LOO counterfactual defense：
     对每篇文档 d_i：删除其 claim（等价于 D\\{d_i}，k=5 逐文档），用与 CCI
     相同的 surrogate 影响公式（答案支持相对 drop）计算 influence(d_i)。
     保留 influence < τ_loo 的文档（阈值在 val 上扫描）。
     这是"逐文档反事实防御"，直接与联盟级删除对比（论文 E8 的前端对照）。

  特征/影响计算复用 coalition.trust（同口径）；两种基线不调用风险学习器。
"""
from __future__ import annotations

import numpy as np

from config import RAG_PROMPT
from coalition.trust import _graph_without, answer_support, trust_weights
from graph import ClaimGraph
from relation.types import Relation, RelationType


# ---------------- B4 NLI-consistency ----------------

def nli_consistency_select(docs, claims, g: ClaimGraph) -> tuple[list, list]:
    """返回 (kept_docs, removed_docs)。最大一致支持簇 = 正边最大连通分量；
    无矛盾边时保留全部；有矛盾但无正边 → 保留支持度最高（信任权重最大）文档集。"""
    has_conflict = len(g.neg_edges) > 0
    if not has_conflict:
        return list(docs), []
    comps = g.coalition_nodes("signed")
    if not comps:
        # 全部孤立（无支持边）但有矛盾 → 无可选簇，保守保留信任权重最高者
        p = trust_weights(g)
        best = max(p, key=p.get) if p else None
        keep = [d for d in docs
                if any(c["claim_id"] == best and c["meta"].get("doc_id") == d["doc_id"]
                       for c in claims)]
        return (keep or list(docs)), [d for d in docs if d not in keep]
    big = max(comps, key=len)
    keep_ids = {c["claim_id"] for c in claims if c["claim_id"] in big}
    keep_docs = [d for d in docs
                 if any(c["claim_id"] in keep_ids and
                        c["meta"].get("doc_id") == d["doc_id"] for c in claims)]
    return keep_docs, [d for d in docs if d not in keep_docs]


# ---------------- B6 per-doc LOO ----------------

def doc_loo_influences(rec: dict) -> dict[str, float]:
    """doc_id → surrogate 影响（D\\{d_i} 相对答案支持 drop，与 CCI 同公式）。"""
    g = rec.get("g")
    if g is None:
        from fp.enrich_features import rebuild_graph
        g = rebuild_graph(rec)
    claims = rec["claims"]
    sims = np.asarray(rec["sims"], dtype=float)
    ids = rec.get("cand_ids", ["cand1", "cand2"])
    w0 = trust_weights(g)
    s0 = answer_support(claims, w0, sims, ids)
    argmax0 = max(s0, key=s0.get)
    denom = s0.get(argmax0, 1.0) or 1.0
    out = {}
    for c in claims:
        did = c["meta"].get("doc_id")
        if did is None:
            continue
        keep_claims = [x for x in claims if x["claim_id"] != c["claim_id"]]
        g2 = _graph_without(g, [c["claim_id"]])
        if not g2.claim_ids:
            out[did] = 0.0
            continue
        w2 = trust_weights(g2)
        idx = {x["claim_id"]: i for i, x in enumerate(claims)}
        sub = sims[[idx[x["claim_id"]] for x in keep_claims]]
        s2 = answer_support(keep_claims, w2, sub, ids)
        imp = max((s0.get(a, 0.0) - s2.get(a, 0.0)) for a in ids)
        out[did] = round(max(0.0, imp) / denom, 4)
    return out


def loo_select(docs, rec, tau_loo: float) -> tuple[list, list]:
    inf = doc_loo_influences(rec)
    kept = [d for d in docs if inf.get(d["doc_id"], 0.0) < tau_loo]
    removed = [d for d in docs if inf.get(d["doc_id"], 0.0) >= tau_loo]
    return kept, removed
