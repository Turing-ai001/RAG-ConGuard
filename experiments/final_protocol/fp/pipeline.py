# -*- coding: utf-8 -*-
"""fp/pipeline.py — 逐 query 证据结构/联盟特征/影响计算（冻结协议 §5）。

与旧 eval_p3 的 collect_features 口径一致（保证可对照），但：
  1) 不依赖 LLM 生成（仅 NLI + judge + BGE），可离线预采集；
  2) 输出结构化记录（docs/claims/edges/records/sims），评估时由 fp.runner 消费；
  3) 记录联盟与文档的映射 + 投毒标注（供诊断/消融，不供运行时使用）。
"""
from __future__ import annotations

import numpy as np

from config import TOP_K
from defense.pipeline3 import claims_from_docs
from graph import ClaimGraph
from coalition.analyze import coalition_features
from coalition.trust import counterfactual_impacts


def collect_features(kb, embedder, extractor, item: dict, qid: str,
                     top_k: int = TOP_K, graph_mode: str = "signed") -> dict:
    """见模块 docstring。

    候选答案（影响公式用）= 与查询 BGE 相似度最高的 2 个 claim 文本
    （target-free：不依赖 gold/攻击答案——论文"without knowing the attack
    target" 与 Phase 3 exact CCI 对齐；2026-08-25 协议 v1.1 修正，旧实现
    用 benchmark 的 correct/incorrect answer 是推理期 oracle 泄漏）。
    """
    docs = kb.search(item["question"], top_k=top_k)
    claims = claims_from_docs(docs)
    edges = extractor.extract_all(claims)
    if graph_mode == "signed_semantic":
        from graph.build import signed_semantic_graph
        g = signed_semantic_graph(claims, edges, embedder, threshold=0.85)
    else:
        g = ClaimGraph(claims, edges)
    coalitions = g.coalition_nodes("signed")
    feats = coalition_features(g, coalitions)
    fmap = {tuple(sorted(f["claims"])): f for f in feats}

    c_embs = embedder.encode([c["text"] for c in claims])
    q_emb = embedder.encode_query(item["question"])
    c_embs = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-9)
    q_embn = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    qsim = c_embs @ q_embn
    order = np.argsort(-qsim)
    cand_idx = list(dict.fromkeys(order.tolist()))[:2]    # 去重保序，取 top-2
    cand_texts = [c["text"] for i, c in enumerate(claims) if i in cand_idx]
    if len(cand_texts) < 2:
        cand_texts += ["the answer cannot be determined from the context"]
    answer_ids = ["cand1", "cand2"]
    a_embs = embedder.encode(cand_texts[:2])
    a_embs = a_embs / (np.linalg.norm(a_embs, axis=1, keepdims=True) + 1e-9)
    sims = c_embs @ a_embs.T

    impacts = counterfactual_impacts(g, coalitions, claims, sims, answer_ids)
    impact_map = {tuple(imp["claims"]): imp for imp in impacts}

    poisoned_ids = {c["claim_id"] for c in claims if c["meta"].get("poisoned")}
    records = []
    for comp in coalitions:
        feat = fmap.get(tuple(sorted(comp)), {})
        imp = impact_map.get(tuple(sorted(comp)), {"impact": 0.0})
        corr = feat.get("external_corroboration", 0.0)
        poisoned_n = sum(1 for cid in comp if cid in poisoned_ids)
        purity = max(poisoned_n / len(comp), 1 - poisoned_n / len(comp)) if comp else 0.0
        records.append({
            "claims": sorted(comp),
            "size": len(comp),
            "purity": round(purity, 3),
            "poisoned_ratio": round(poisoned_n / len(comp), 3) if comp else 0.0,
            "cohesion": feat.get("cohesion", 0.0),
            "external_corroboration": corr,
            "external_refute": feat.get("external_refute", 0.0),
            "impact": float(imp["impact"]),
            "control": round(float(imp["impact"]) * (1 - min(corr, 1.0)), 4),
        })

    return {"docs": docs, "claims": claims, "edges": edges, "g": g,
            "sims": sims, "cand_ids": answer_ids, "q_emb": q_embn,
            "coalitions": records}


def claim_to_doc_map(claims: list[dict]) -> dict[str, str]:
    return {c["claim_id"]: c["meta"].get("doc_id") for c in claims}
