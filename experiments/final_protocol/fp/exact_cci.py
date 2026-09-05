# -*- coding: utf-8 -*-
"""fp/exact_cci.py — Exact CCI（生成分布级反事实影响，Phase 3）。

定义（论文 M6 形式化）：
    CCI_exact(C) = (1/H) Σ_t JSD( p_t^full ∥ p_t^{-C} )
    p_t^full     = P(x_t | x_<t, q, D)
    p_t^{-C}     = P(x_t | x_<t, q, D\\C)

对齐保证：先以 full context D 生成 H-token 探针序列 x_1..x_H（T=0）；
再把同一前缀 x_1..x_{t-1} 分别 teacher-force 进 full 与 D\\C 两个上下文
（llm_hf_backend.probe_logits），逐位置比较——两次 forward 共享完全相同的
前缀，JSD 只反映证据集变化。

成本：每 (query, coalition) 2 次 forward；每 query 1 次 16-token 解码。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from config import RAG_PROMPT


def jsd_pp(p1: np.ndarray, p2: np.ndarray) -> float:
    """两个离散分布的 Jensen-Shannon divergence（log2 底）。"""
    m = 0.5 * (p1 + p2)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p1 > 0, p1 * np.log2(p1 / m), 0.0)
        b = np.where(p2 > 0, p2 * np.log2(p2 / m), 0.0)
    return float(0.5 * (a.sum() + b.sum()))


def run_exact_cci(rec: dict, llm, H: int = 16, min_size: int = 2) -> dict:
    """对单 query 的每个联盟计算 exact CCI + surrogate（并排记录）。

    rec: fp.pipeline.collect_features 的结构（docs/claims/coalitions/g/sims）。
    返回 {"probe": str, "coalitions": [
        {claims, size, exact_cci, surrogate_impact, surrogate_control,
         poisoned_ratio, purity} ...]}（查询内按 rec 图的 signed 联盟）。
    """
    from coalition.analyze import coalition_features
    from coalition.trust import counterfactual_impacts
    from fp.enrich_features import rebuild_graph

    docs, claims, q = rec["docs"], rec["claims"], rec["question"]
    g = rebuild_graph(rec)
    full_prompt = RAG_PROMPT.replace("[question]", q).replace(
        "[context]", "\n".join(d["text"] for d in docs))

    # ---- 1) 探针序列（full context, T=0）----
    raw = llm.complete(full_prompt, max_new_tokens=H)
    T = llm._tokenizer
    ids = T(raw, return_tensors="pt", truncation=True, max_length=4096)[
        "input_ids"][0].tolist()
    probe_ids = ids[-H:]            # 取后 H 个 token 作为稳定前缀

    # ---- 2) 每联盟：full / minus-C 两次 forward（相同前缀)----
    out = []
    for comp in g.coalition_nodes("signed"):
        if len(comp) < min_size:
            continue
        cdoc_ids = {c["meta"].get("doc_id") for c in claims
                    if c["claim_id"] in comp}
        minus_docs = [d for d in docs if d["doc_id"] not in cdoc_ids]
        if not minus_docs:
            continue
        minus_prompt = RAG_PROMPT.replace("[question]", q).replace(
            "[context]", "\n".join(d["text"] for d in minus_docs))
        L_full = llm.probe_logits(full_prompt, probe_ids)
        L_minus = llm.probe_logits(minus_prompt, probe_ids)
        n_t = min(L_full.shape[0], L_minus.shape[0], H)
        jsds = [jsd_pp(
            F.softmax(L_full[t], dim=-1).cpu().numpy().astype(np.float64),
            F.softmax(L_minus[t], dim=-1).cpu().numpy().astype(np.float64))
            for t in range(n_t)]
        exact = float(np.mean(jsds)) if jsds else 0.0

        # ---- 3) surrogate（与 eval_p3 同口径，同 query 记录对照）----
        feats = coalition_features(g, [comp])
        f = feats[0] if feats else {}
        from coalition.trust import counterfactual_impacts as _ci
        imp = _ci(g, [comp], claims, np.asarray(rec["sims"], dtype=float),
                  rec.get("cand_ids", ["cand1", "cand2"]))
        imp_i = imp[0] if imp else {"impact": 0.0}
        corr = f.get("external_corroboration", 0.0)
        poisoned_n = sum(1 for cid in comp
                         if any(c["claim_id"] == cid and
                                c["meta"].get("poisoned") for c in claims))
        size = len(comp)
        out.append({
            "claims": sorted(comp),
            "size": size,
            "exact_cci": round(exact, 6),
            "surrogate_impact": round(float(imp_i["impact"]), 6),
            "surrogate_control": round(float(imp_i["impact"]) *
                                       (1 - min(corr, 1.0)), 6),
            "poisoned_ratio": round(poisoned_n / size, 3),
            "purity": round(max(poisoned_n / size, 1 - poisoned_n / size), 3),
        })
    return {"probe": raw, "coalitions": out}
