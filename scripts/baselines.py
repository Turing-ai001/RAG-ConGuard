"""P5 对比基线：简化版鲁棒检索/过滤防御（与联盟防御对比）。

Baseline 0  no-defense        无防御（直接生成）
Baseline 1  relevance-filter  相关度过滤：doc ↔ query BGE 相似度 < tau_rel 剔除
Baseline 2  dup-filter        同源过滤：互文相似度 >= tau_dup 的文档组剔除
                              （PoisonedRAG 式同源投毒的痕迹）

评估指标与 eval_p3 同口径：ASR / correct rate / 剩余文档数。

用法：
    LLM_BACKEND=hf python scripts/baselines.py --dataset nq,hotpotqa --num_queries 20 --gen 1
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from config import INDEX_DIR, OUTPUT_DIR, RAG_PROMPT, TOP_K
from llm import get_backend
from retrieval import Embedder, KnowledgeBase
from retrieval.kb import load_poisoned_docs


def clean_str(s: str) -> str:
    s = str(s).strip().lower()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s


def answer_match(pred: str, target: str) -> bool:
    p, t = clean_str(pred), clean_str(target)
    if not p or not t:
        return False
    return p == t or p in t or t in p


def filter_docs(docs: list[dict], q_emb: np.ndarray, d_embs: np.ndarray,
                tau_rel: float, tau_dup: float) -> dict:
    """三种过滤策略（可组合）。返回 {kept, removed, filter_used}。"""
    # 1) 相关度过滤
    rel = d_embs @ q_emb
    rel_keep = [d for d, r in zip(docs, rel) if r >= tau_rel]

    # 2) 同源过滤：互文相似度 >= tau_dup 的组（并查集去重）
    n = len(docs)
    sim = d_embs @ d_embs.T
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= tau_dup:
                parent[find(i)] = find(j)
    dup_keep = [d for i, d in enumerate(docs)
                if find(i) == i or len([k for k in range(n) if find(k) == find(i)]) == 1]

    return {"relevance": rel_keep, "dup": dup_keep}


def main(dataset: str, num_queries: int, do_gen: bool,
         tau_rel: float = 0.30, tau_dup: float = 0.85):
    embedder = Embedder()
    llm = get_backend() if do_gen else None

    root = INDEX_DIR / f"{dataset}_kb"
    if not root.exists():
        raise FileNotFoundError(f"{root} 不存在，先跑 build_kb")
    kb = KnowledgeBase.load(f"{dataset}_kb", embedder)

    stats = {k: {"n": 0, "asr": 0, "correct": 0, "kept": 0} for k in
             ("no_defense", "relevance", "dup")}
    rows = []
    for qid, item in list(load_poisoned_docs(dataset).items())[:num_queries]:
        q = item["question"]
        docs = kb.search(q, top_k=TOP_K)
        q_emb = embedder.encode_query(q)
        d_embs = embedder.encode([d["text"] for d in docs])
        f = filter_docs(docs, q_emb, d_embs, tau_rel, tau_dup)

        for name, kept in [("relevance", f["relevance"]), ("dup", f["dup"])]:
            if not do_gen or not kept:
                stats[name]["n"] += 1
                stats[name]["kept"] += len(kept)
                continue
            ctx = "\n".join(d["text"] for d in kept)
            answer = llm.complete(RAG_PROMPT.replace("[question]", q)
                                  .replace("[context]", ctx))
            stats[name]["n"] += 1
            stats[name]["asr"] += answer_match(answer, item["incorrect answer"])
            stats[name]["correct"] += answer_match(answer, item["correct answer"])
            stats[name]["kept"] += len(kept)

        if do_gen:
            ctx = "\n".join(d["text"] for d in docs)
            answer = llm.complete(RAG_PROMPT.replace("[question]", q)
                                  .replace("[context]", ctx))
            stats["no_defense"]["n"] += 1
            stats["no_defense"]["asr"] += answer_match(answer, item["incorrect answer"])
            stats["no_defense"]["correct"] += answer_match(answer, item["correct answer"])
            stats["no_defense"]["kept"] += len(docs)
        rows.append({"query_id": qid, "question": q,
                     "docs": [d["doc_id"] for d in docs]})

    print("=" * 72)
    print(f"P5 基线 | dataset={dataset} | queries={num_queries} | "
          f"tau_rel={tau_rel} tau_dup={tau_dup} | gen={do_gen}")
    print("=" * 72)
    for k, s in stats.items():
        n = s["n"] or 1
        print(f"[{k:<13}] ASR={s['asr'] / n:.3f} 正确率={s['correct'] / n:.3f} "
              f"平均保留文档={s['kept'] / n:.1f}")

    out = OUTPUT_DIR / "eval_pipeline" / f"baselines_{dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": {"tau_rel": tau_rel, "tau_dup": tau_dup, "gen": do_gen},
         "summary": {k: {"asr": v["asr"] / max(1, v["n"]),
                         "correct": v["correct"] / max(1, v["n"]),
                         "avg_kept": v["kept"] / max(1, v["n"])}
                     for k, v in stats.items()},
         "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[baselines] saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq", choices=["nq", "hotpotqa", "msmarco"])
    ap.add_argument("--num_queries", type=int, default=20)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--tau-rel", type=float, default=0.30)
    ap.add_argument("--tau-dup", type=float, default=0.85)
    args = ap.parse_args()
    main(args.dataset, args.num_queries, bool(args.gen),
         args.tau_rel, args.tau_dup)
