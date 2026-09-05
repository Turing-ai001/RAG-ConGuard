"""联盟级反事实影响（P2 后半）：图上的答案支持聚合 + 阻断分析。

f(G)（图聚合分数）：证据图上每个 claim 的信任权重 p_c（传播：被支持→升，
被反驳→压）与候选答案语义匹配的加权和：
    s_G(A) = Σ_c p_c · max(0, sim(c, A))

反事实影响：删除联盟 C（节点+关联边）后重新传播得到 f(G \\ C)：
    impact(C) = max_A [ s_G(A) − s_{G\\C}(A) ] / s_G(A*)          （相对，跨 query 可比）
异常控制力：影响 × 佐证缺失度：
    control(C) = impact × (1 − min(external_corroboration, 1))

计算代价：每 query 图仅 15–25 节点、联盟 2–5 个，直接精确计算
（对每个联盟重建一次图），无需枚举子集或近似。
"""
from __future__ import annotations

import numpy as np

from config import P2_MIN_COALITION_SIZE, P2_REFUTE_WEIGHT, P2_TRUST_ITERS
from graph.build import ClaimGraph


# ---------- 信任传播 ----------
def trust_weights(g: ClaimGraph, refute_weight: float = P2_REFUTE_WEIGHT,
                  iters: int = P2_TRUST_ITERS) -> dict[str, float]:
    """图内信任传播。

    p_{t+1}(c) = (0.5 + Σ_{s→c 正边} p_t(s)) / (1 + refute_weight × Σ_{s→c 负边} p_t(s))

    被多个"支持边"背书 → 升（上限由分母控制）；被反驳 → 压（refute_weight > 1 加重）。
    """
    pos_in: dict[str, list[str]] = {n: [] for n in g.claim_ids}
    neg_in: dict[str, list[str]] = {n: [] for n in g.claim_ids}
    for s, d in g.pos_edges:
        pos_in.setdefault(d, []).append(s)
    for s, d in g.neg_edges:
        neg_in.setdefault(d, []).append(s)

    p = {n: 1.0 for n in g.claim_ids}
    for _ in range(max(1, iters)):
        new_p = {}
        for c in g.claim_ids:
            sup = sum(p.get(s, 1.0) for s in pos_in.get(c, []))
            ref = sum(p.get(s, 1.0) for s in neg_in.get(c, []))
            new_p[c] = (0.5 + sup) / (1.0 + refute_weight * ref)
        p = new_p
    return p


# ---------- 答案支持 ----------
def answer_support(claims: list[dict], weights: dict[str, float],
                   sims: np.ndarray, answer_ids: list[str]) -> dict[str, float]:
    """s_G(A) = Σ_c p_c · max(0, sim(c, A))。

    sims: (n_claims, n_answers) BGE 余弦矩阵（已 clip 负值由未归一化 p 控制）。
    """
    ids = [c["claim_id"] for c in claims]
    scores = {}
    for j, aid in enumerate(answer_ids):
        w = np.array([weights.get(i, 1.0) for i in ids])
        scores[aid] = float(np.sum(w * np.maximum(sims[:, j], 0.0)))
    return scores


def _graph_without(g: ClaimGraph, removed: list[str]) -> ClaimGraph:
    """删除节点集合后的子图（节点+关联边全部移除），重新构造。"""
    removed_set = set(removed)
    keep_ids = [c for c in g.claim_ids if c not in removed_set]
    keep_claims = [c for c in g.claims if c["claim_id"] not in removed_set]
    keep_rels = [r for r in g.relations
                 if r.src not in removed_set and r.dst not in removed_set]
    return ClaimGraph(keep_claims, keep_rels)


# ---------- 反事实影响 ----------
def counterfactual_impacts(g: ClaimGraph, coalitions: list[set[str]],
                           claims: list[dict], sims: np.ndarray,
                           answer_ids: list[str],
                           min_size: int = P2_MIN_COALITION_SIZE) -> list[dict]:
    """对每个联盟计算反事实影响。

    返回 [{claims, size, impact, control, s_before, s_after, argmax_before, argmax_after}]。
    impact 仅在 |C| >= min_size 且删除后图仍有节点时计算，否则 0。
    """
    w0 = trust_weights(g)
    s0 = answer_support(claims, w0, sims, answer_ids)
    argmax0 = max(s0, key=s0.get)

    out = []
    for comp in coalitions:
        nodes = sorted(comp)
        if len(nodes) < min_size:
            continue
        g2 = _graph_without(g, nodes)
        if not g2.claim_ids:
            continue
        w2 = trust_weights(g2)
        keep_claims = [c for c in claims if c["claim_id"] in g2.claim_ids]
        idx = {c["claim_id"]: i for i, c in enumerate(claims)}
        sub_sims = sims[[idx[c["claim_id"]] for c in keep_claims]]
        s2 = answer_support(keep_claims, w2, sub_sims, answer_ids)

        denom = s0.get(argmax0, 1.0) or 1.0
        impact = max((s0.get(a, 0.0) - s2.get(a, 0.0)) for a in answer_ids)
        impact /= denom                    # 相对影响 ∈ [0, 1]（跨 query 可比）
        impact = max(0.0, impact)
        argmax_after = max(s2, key=s2.get)
        out.append({
            "claims": nodes,
            "size": len(nodes),
            "impact": round(impact, 4),
            "s_before": round(float(s0.get(argmax_after, 0.0)), 4),
            "s_after": round(float(s2.get(argmax_after, 0.0)), 4),
            "argmax_before": argmax0,
            "argmax_after": argmax_after,
            # control 强度由 eval 脚本叠加佐证度（analyze 输出）后计算
        })
    return out
