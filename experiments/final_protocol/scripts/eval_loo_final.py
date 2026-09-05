# -*- coding: utf-8 -*-
"""E8 最终口径重放：逐文档 LOO 归因 vs 联盟级删除（target-free 候选答案）。

冻结协议版本（test split，富化特征，非 oracle 基准答案）：
    per-doc influence(d_i)：删除 D\\{d_i} 后候选答案集（top-2 查询相关 claim）的
        相对支持 drop（与 coalition.trust 同公式）
    coalitional removal impact：删除整联盟后同上
按投毒组规模分组报告（2/3/4 篇），口径与论文表 4 对齐但为最终协议版本。
只读特征文件，无生成、无 GPU。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/RAG")
sys.path.insert(0, "/root/autodl-tmp/RAG/experiments/final_protocol")

import numpy as np

from fp.baselines_def import doc_loo_influences
from fp.enrich_features import rebuild_graph
from fp.runner import load_features, _features_path

base = Path("/root/autodl-tmp/RAG/experiments/final_protocol/results/p1")

bidx = defaultdict(list)      # size -> [loo, cci]
rows = []
n_groups = 0
for ds in ("nq", "hotpotqa", "msmarco"):
    for rec in load_features(_features_path(base, ds, "test")):
        g = rebuild_graph(rec)
        inf = doc_loo_influences(rec)      # doc_id -> per-doc influence
        # 联盟级影响（与整联盟删除等价）
        from coalition.trust import counterfactual_impacts, trust_weights, answer_support
        claims, sims = rec["claims"], np.asarray(rec["sims"], dtype=float)
        ids = rec.get("cand_ids", ["cand1", "cand2"])
        w0 = trust_weights(g)
        s0 = answer_support(claims, w0, sims, ids)
        argmax0 = max(s0, key=s0.get)
        denom = s0.get(argmax0, 1.0) or 1.0
        for comp in g.coalition_nodes("signed"):
            if len(comp) < 2:
                continue
            # 组内投毒文档数
            po = sum(1 for cid in comp
                     if any(c["claim_id"] == cid and c["meta"].get("poisoned")
                            for c in claims))
            if po < 2:
                continue
            cdoc_ids = {c["meta"].get("doc_id") for c in claims
                        if c["claim_id"] in comp}
            loo = [inf.get(did, 0.0) for did in cdoc_ids if did]
            loo_mean = float(np.mean(loo)) if loo else 0.0
            keep_claims = [c for c in claims if c["claim_id"] not in comp]
            if not keep_claims:
                continue
            from coalition.trust import _graph_without
            g2 = _graph_without(g, list(comp))
            if not g2.claim_ids:
                continue
            w2 = trust_weights(g2)
            idx = {c["claim_id"]: i for i, c in enumerate(claims)}
            sub = sims[[idx[c["claim_id"]] for c in keep_claims]]
            s2 = answer_support(keep_claims, w2, sub, ids)
            cci = max((s0.get(a, 0.0) - s2.get(a, 0.0)) for a in ids) / denom
            bidx[len(comp if len(comp) <= 5 else 4)].append((loo_mean, max(0.0, cci)))
            n_groups += 1

out = {}
for sz, pairs in sorted(bidx.items(), key=lambda kv: kv[0]):
    l = [p[0] for p in pairs]
    c = [p[1] for p in pairs]
    out[f"{sz} docs"] = {"n": len(pairs),
                         "avg_per_doc_loo": round(float(np.mean(l)), 3),
                         "avg_coalitional": round(float(np.mean(c)), 3),
                         "ratio": round(float(np.mean(c) / np.mean(l)), 2)
                         if np.mean(l) > 0 else None}
print(json.dumps(out, ensure_ascii=False, indent=1))
print("total poisoned groups (>=2 poisoned):", n_groups)
(base / "eval_loo_final.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
