# -*- coding: utf-8 -*-
"""fp/enrich_features.py — 给已收集的 features JSONL 补写 edges/sims/cand_ids。

原因：v1.x 特征文件仅存 docs/claims/coalitions；exact CCI、B4、B6 需要
图边与答案候选相似度。增量重算（每 query 一次 NLI+encode，几千条量级），
只对指定的 (run-dir, dataset, split) 富化，覆写回原文件（保留原字段 +
新字段；重复跑幂等）。

用法：
    python -m fp.enrich_features --run-dir results/p1 --splits val,test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fp.pipeline import ClaimGraph  # noqa: F401 (re-export for callers)


def relation_to_dict(r) -> dict:
    return {"src": r.src, "dst": r.dst, "type": r.type.name,
            "score": float(r.score), "provenance": r.provenance}


def relation_from_dict(d: dict):
    from relation.types import Relation, RelationType
    return Relation(d["src"], d["dst"], RelationType[d["type"]],
                    d["score"], provenance=d.get("provenance", "nli"))


def rebuild_graph(rec: dict):
    rels = [relation_from_dict(e) for e in rec.get("edges", [])]
    return ClaimGraph(rec["claims"], rels)


def enrich_rec(kb, embedder, extractor, rec: dict) -> dict:
    import numpy as np
    docs, claims = rec["docs"], rec["claims"]
    edges = extractor.extract_all(claims)
    rec["edges"] = [relation_to_dict(e) for e in edges]
    c_embs = embedder.encode([c["text"] for c in claims])
    q_emb = embedder.encode_query(rec["question"])
    c_embs = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-9)
    q_embn = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    qsim = c_embs @ q_embn
    order = np.argsort(-qsim)
    cand_idx = list(dict.fromkeys(order.tolist()))[:2]
    cand_texts = [c["text"] for i, c in enumerate(claims) if i in cand_idx]
    if len(cand_texts) < 2:
        cand_texts += ["the answer cannot be determined from the context"]
    a_embs = embedder.encode(cand_texts[:2])
    a_embs = a_embs / (np.linalg.norm(a_embs, axis=1, keepdims=True) + 1e-9)
    rec["sims"] = (c_embs @ a_embs.T).tolist()
    rec["cand_ids"] = ["cand1", "cand2"]
    rec["q_emb"] = q_embn.tolist()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/p1")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--datasets", default="nq,hotpotqa,msmarco")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    from relation import LLMJudge, NLIScorer, RelationExtractor
    from llm import get_backend
    from retrieval import Embedder, KnowledgeBase
    from fp.runner import _features_path, load_features

    embedder, nli = Embedder(), NLIScorer()
    llm = get_backend()
    extractor = RelationExtractor(nli, LLMJudge(llm))

    datasets = [d.strip() for d in args.datasets.split(",")]
    for split in [s.strip() for s in args.splits.split(",")]:
        for ds in datasets:
            p = _features_path(run_dir, ds, split)
            if not p.exists():
                print(f"skip {ds}/{split}: no file", flush=True)
                continue
            recs = load_features(p)
            kb = KnowledgeBase.load(f"{ds}_kb", embedder)
            t0 = time.perf_counter()
            out = []
            for rec in recs:
                r = enrich_rec(kb, embedder, extractor, rec)
                r.pop("g", None)          # 不落盘对象
                out.append(r)
            with open(p, "w", encoding="utf-8") as f:
                for r in out:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[enrich] {ds}/{split} n={len(out)} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
