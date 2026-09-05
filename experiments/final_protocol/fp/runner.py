# -*- coding: utf-8 -*-
"""fp/runner.py — 统一评估入口（冻结协议）。

用法（在 experiments/final_protocol/ 目录）：
    python -m fp.runner --stage features --split train,val --run-dir results/p1
    python -m fp.runner --stage train --run-dir results/p1
    python -m fp.runner --stage eval --split test --methods B0,G0 --run-dir results/p1

方法 ID（冻结协议 §6）：
    B0 vanilla | B1 rel_filter(tau_rel) | B2 dup_filter(tau_dup)
    G0/G1/G2 = conguard + selector s0/s1/s2（S0/S1 参数 tau，S2 参数 lam）
eval 以同一 split 依次跑所有 methods（生成 + 匹配），输出 predictions.jsonl
（逐 query 互斥主状态）+ metrics.json（分方法 + 检索 + paired bootstrap）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
os.environ.setdefault("HF_OFFLINE", "1")

from config import RAG_PROMPT

from fp.data import clean_queries, iter_split_queries
from fp.io import (RunResult, auc, main_state, metrics_from_predictions,
                   paired_bootstrap)
from fp.pipeline import collect_features
from fp.selectors import SELECTORS

METHODS = {
    "B0": ("vanilla", {}),
    "B1": ("rel_filter", {"tau_rel": 0.30}),
    "B2": ("dup_filter", {"tau_dup": 0.85}),
    "B4": ("nli_consistency", {}),
    "B5": ("trustrag", {}),
    "B6": ("doc_loo", {"tau_loo": 0.10}),
    "G0": ("conguard", {"selector": "s0", "tau": 0.25}),
    "G1": ("conguard", {"selector": "s1", "tau": 0.25}),
    "G2": ("conguard", {"selector": "s2", "lam": 0.6}),
}


# ---------------- features ----------------

def _features_path(run_dir: Path, ds: str, split: str) -> Path:
    return run_dir / f"features_{ds}_{split}.jsonl"


def collect_ds_split(kb, embedder, extractor, ds, split, out_path: Path,
                     clean: bool = False, graph_mode: str = "signed"):
    t0 = time.perf_counter()
    n = 0
    recs = []
    if not clean:
        for _, qid, item in iter_split_queries([ds], split):
            r = collect_features(kb, embedder, extractor, item, qid,
                                 graph_mode=graph_mode)
            recs.append({
                "dataset": ds, "query_id": qid, "question": item["question"],
                "gold": item["correct answer"],
                "attack_target": item["incorrect answer"],
                **{k: r[k] for k in ("docs", "claims", "coalitions")},
            })
    else:
        for q in clean_queries(ds, split):
            r = collect_features(kb, embedder, extractor,
                                 {"question": q["question"]}, q["id"],
                                 graph_mode=graph_mode)
            recs.append({
                "dataset": ds, "query_id": q["id"], "question": q["question"],
                "gold": q.get("gold"), "attack_target": None,
                **{k: r[k] for k in ("docs", "claims", "coalitions")},
            })
    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[features] {ds}/{split} n={len(recs)} -> {out_path} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)


def load_features(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8")
            .split("\n") if line.strip()]


# ---------------- train ----------------

def _asr_pool(feats: list[dict]) -> tuple[list[dict], list[int]]:
    """训练池。标签（2026-08-25 修正）：poisoned_ratio ≥ 0.8 = 投毒联盟。

    注意修正：旧实现用 purity=max(投毒比, 正常比)——all-benign 联盟 purity
    同样 =1.0 → 被误标为正例（这是 refusal=100% 的根因之一）。
    正确标签按协议 M7：≥80% 成员投毒 = 正例；all-benign = 0。
    """
    from fp.selectors import is_coalition
    pool, labels = [], []
    for rec in feats:
        for c in rec["coalitions"]:
            if not is_coalition(c):
                continue          # v1.2：仅"最终联盟"进训练池
            pool.append({**c, "dataset": rec["dataset"],
                         "query_id": rec["query_id"]})
            labels.append(1 if c.get("poisoned_ratio", 0.0) >= 0.8 else 0)
    return pool, labels


def stage_train(args, run_dir: Path):
    from defense.risk import LearnedRisk, feature_matrix
    feats = []
    for ds in args.datasets.strip().split(","):
        p = _features_path(run_dir, ds.strip(), "train")
        if p.exists():
            feats += load_features(p)
    pool, labels = _asr_pool(feats)
    # 干净负例池（协议 §2）：clean_trn 查询的联盟全部视为良性（label=0）
    clean_feats = []
    for ds in args.datasets.strip().split(","):
        p = _features_path(run_dir, ds.strip(), "clean_trn")
        if p.exists():
            clean_feats += load_features(p)
    from fp.selectors import is_coalition as _ic
    for rec in clean_feats:
        for c in rec["coalitions"]:
            if not _ic(c):
                continue
            pool.append({**c, "dataset": rec["dataset"],
                         "query_id": rec["query_id"]})
            labels.append(0)
    risk = LearnedRisk(random_state=args.seed)
    risk.fit(pool, labels)
    ckpt = run_dir / "risk_learner.pkl"
    risk.save(str(ckpt))
    scores = risk.predict_proba(feature_matrix(pool)).tolist()
    m = {"train_pool": len(pool), "train_auc": auc(scores, labels),
         "train_pos": sum(labels), "train_neg": len(labels) - sum(labels)}
    (run_dir / "risk_train_metrics.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] pool={len(pool)} pos={m['train_pos']} "
          f"auc={m['train_auc']:.3f} -> {ckpt}", flush=True)


# ---------------- method 执行 ----------------

def _norm_claim_embs(embedder, claims):
    e = embedder.encode([c["text"] for c in claims])
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


def _ctx(ds):
    return "\n".join(x["text"] for x in ds)


def _complete(llm, q: str, docs) -> str:
    return llm.complete(RAG_PROMPT.replace("[question]", q)
                        .replace("[context]", _ctx(docs)))


def run_method(mid: str, rec: dict, llm, embedder, risk_fn=None) -> dict:
    docs, claims, coalitions, q = rec["docs"], rec["claims"], rec["coalitions"], rec["question"]
    mtype, params = METHODS[mid]
    detected = []

    if mtype == "vanilla":
        ans = _complete(llm, q, docs)
        return {"answer": ans, "kept": docs, "removed": [], "refused": False,
                "detected": detected}

    if mtype == "rel_filter":
        c_embs, q_emb = _norm_claim_embs(embedder, claims), embedder.encode_query(q)
        rel = c_embs @ q_emb
        keep = [d for d, r in zip(docs, rel) if r >= params["tau_rel"]]
        ans = _complete(llm, q, keep)
        return {"answer": ans, "kept": keep, "removed": [d for d in docs
                                                         if d not in keep],
                "refused": False, "detected": detected}

    if mtype == "dup_filter":
        c_embs = _norm_claim_embs(embedder, claims)
        n = len(docs)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(n):
            for j in range(i + 1, n):
                if c_embs[i] @ c_embs[j] >= params["tau_dup"]:
                    parent[find(i)] = find(j)
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        drop = set()
        for _, idxs in groups.items():
            if len(idxs) > 1:
                drop.update(idxs[1:])     # 同源组剔除（保留第一个 = 旧口径）
        keep = [d for i, d in enumerate(docs) if i not in drop]
        ans = _complete(llm, q, keep)
        return {"answer": ans, "kept": keep,
                "removed": [d for i, d in enumerate(docs) if i in drop],
                "refused": False, "detected": detected}

    if mtype == "conguard":
        if risk_fn is None:
            raise RuntimeError("conguard 需要 risk learner（先 stage train）")
        sel = SELECTORS[params["selector"]](
            docs, claims, coalitions, risk_fn,
            **{k: v for k, v in params.items() if k != "selector"})
        detected = sel["removed_coalitions"]
        if sel["refused"]:
            return {"answer": "I don't know", "kept": [], "removed": sel["removed_docs"],
                    "refused": True, "detected": detected}
        ans = _complete(llm, q, sel["kept_docs"])
        return {"answer": ans, "kept": sel["kept_docs"],
                "removed": sel["removed_docs"], "refused": False,
                "detected": detected}

    if mtype == "nli_consistency":
        from fp.baselines_def import nli_consistency_select
        from fp.enrich_features import rebuild_graph
        keep, removed = nli_consistency_select(docs, claims,
                                               rebuild_graph(rec))
        if not keep:
            return {"answer": "I don't know", "kept": [], "removed": removed,
                    "refused": True, "detected": []}
        return {"answer": _complete(llm, q, keep), "kept": keep,
                "removed": removed, "refused": False, "detected": []}

    if mtype == "doc_loo":
        from fp.baselines_def import loo_select
        from fp.enrich_features import rebuild_graph
        rec2 = dict(rec)
        rec2["g"] = rebuild_graph(rec)     # baselines_def 的 impact 需 g
        keep, removed = loo_select(docs, rec2, params["tau_loo"])
        if not keep:
            return {"answer": "I don't know", "kept": [], "removed": removed,
                    "refused": True, "detected": []}
        ans = _complete(llm, q, keep)
        return {"answer": ans, "kept": keep, "removed": removed,
                "refused": False, "detected": []}

    if mtype == "trustrag":
        from fp.trustrag_b5 import trustrag_answer
        doc_embs = embedder.encode([d["text"] for d in docs])
        out = trustrag_answer(q, docs, doc_embs, llm)
        return {"answer": out["answer"], "kept": out["kept"],
                "removed": out["removed"], "refused": out["refused"],
                "detected": []}

    raise ValueError(mid)


# ---------------- eval ----------------

def stage_eval(args, run_dir: Path, datasets: list[str], split: str,
               llm, embedder, risk_fn=None):
    methods = [m.strip() for m in args.methods.split(",")]
    import os as _os
    tg = _os.environ.get("HF_MODEL", "qwen").replace("/", "-").split("-")[-1]
    run_result = RunResult(run_dir,
                           f"eval_{split}_{'_'.join(methods)}_{tg}")
    preds = []
    for ds in datasets:
        feats = load_features(_features_path(run_dir, ds, split))
        for rec in feats:
            for mid in methods:
                t0 = time.perf_counter()
                try:
                    out = run_method(mid, rec, llm, embedder, risk_fn)
                except Exception as e:
                    print(f"[WARN] {mid}/{ds}/{rec['query_id']}: {e}", flush=True)
                    out = {"answer": "", "kept": [], "removed": [],
                           "refused": False, "detected": []}
                state = main_state(out["answer"], rec.get("gold") or "",
                                   rec.get("attack_target") or "")
                preds.append({
                    "query_id": rec["query_id"], "dataset": ds,
                    "model": "qwen2.5-7b-instruct",
                    "attack": "targeted_5per_query",
                    "method": mid,
                    "retrieved_docs": [d["doc_id"] for d in rec["docs"]],
                    "poison_labels": [":poisoned:" in d["doc_id"]
                                      for d in rec["docs"]],
                    "detected_coalition": out["detected"],
                    "removed_docs": [d["doc_id"] for d in out["removed"]],
                    "final_context": "\n".join(d["text"] for d in out["kept"]),
                    "answer": out["answer"],
                    "gold_answer": rec.get("gold"),
                    "attack_target": rec.get("attack_target"),
                    "attack_success": state == "attack_success",
                    "correct": state == "correct",
                    "refused": out["refused"] or state == "refusal",
                    "state": state,
                    "latency_sec": round(time.perf_counter() - t0, 2),
                })

    metric_map = {mid: metrics_from_predictions(
        [p for p in preds if p["method"] == mid]) for mid in methods}
    if split.startswith("clean"):
        for mid in methods:
            sub = [p for p in preds if p["method"] == mid]
            mf_all, ref = [], 0
            for p in sub:
                kept = [d for d in p["retrieved_docs"] if d not in p["removed_docs"]]
                benign = [d for d in p["retrieved_docs"]
                          if ":poisoned:" not in d]
                mf = (len(benign) - sum(1 for d in benign if d in kept)) \
                    / len(benign) if benign else 0.0
                mf_all.append(mf)
                ref += 1 if p["refused"] else 0
            metric_map[mid].update({
                "clean_refusal": round(ref / len(sub), 4) if sub else None,
                "benign_mis_filter": round(sum(mf_all) / len(mf_all), 4)
                if mf_all else None,
            })

    ret = []
    for ds in datasets:
        for rec in load_features(_features_path(run_dir, ds, split)):
            labels = [":poisoned:" in d["doc_id"] for d in rec["docs"]]
            ret.append({"n_p": sum(labels), "hit": any(labels)})
    n = len(ret)
    metrics = {
        "split": split,
        "methods": metric_map,
        "retrieval": {
            "asr@5": round(sum(r["hit"] for r in ret) / n, 4) if n else None,
            "avg_poisoned_top5": round(
                sum(r["n_p"] for r in ret) / n, 3) if n else None,
            "n_queries": n,
        },
    }
    pb = {}
    a = [p for p in preds if p["method"] == "B0"]
    if a:
        for mid in methods:
            b = [p for p in preds if p["method"] == mid]
            if b:
                pb[mid] = {k: paired_bootstrap(a, b, k, seed=args.seed)
                           for k in ("asr", "accuracy", "refusal")}
    metrics["paired_bootstrap_vs_B0"] = pb
    run_result.save({"protocol": "final_protocol v1.0", "split": split,
                     "methods": methods, "datasets": datasets,
                     "seed": args.seed}, preds, metrics)
    print(f"[eval] {split} methods={methods} "
          f"metrics={json.dumps(metric_map, ensure_ascii=False)[:300]}",
          flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="features", choices=["features", "train",
                                                           "eval", "all"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods", default="B0,G0")
    ap.add_argument("--datasets", default="nq,hotpotqa,msmarco")
    ap.add_argument("--run-dir", default="results/p1")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--g-param", default=None,
                    help='JSON 覆盖首个 G 方法的参数，如 {"tau": 0.2} 或 {"lam": 0.6}')
    ap.add_argument("--methods-params", default=None,
                    help='JSON 按方法覆盖参数，如 {"G0": {"tau": 0.25}, "G2": {"lam": 0.2}}')
    ap.add_argument("--graph-mode", default="signed",
                    choices=["signed", "signed_semantic"])
    ap.add_argument("--risk-kind", default="mlp", choices=["mlp", "rule"],
                    help="mlp=训练学习器（默认）；rule=规则风险（消融用）")
    args = ap.parse_args()
    overrides = {}
    if args.g_param:
        overrides = json.loads(args.g_param)
    if args.methods_params:
        overrides.update(json.loads(args.methods_params))
    for mid, par in overrides.items():
        if mid in METHODS:
            mtype, params = METHODS[mid]
            METHODS[mid] = (mtype, {**params, **par})
        else:
            for m2 in args.methods.split(","):
                m2 = m2.strip()
                if m2.startswith(mid) or mid.startswith(m2):
                    mtype, params = METHODS[m2]
                    METHODS[m2] = (mtype, {**params, **par})
                    break

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",")]

    from relation import LLMJudge, NLIScorer, RelationExtractor
    from llm import get_backend
    from retrieval import Embedder, KnowledgeBase

    embedder = Embedder()
    nli = NLIScorer()
    llm = get_backend()
    judge = LLMJudge(llm)
    extractor = RelationExtractor(nli, judge)

    if args.stage in ("features", "all"):
        for split in [s.strip() for s in args.split.split(",")]:
            for ds in datasets:
                kb = KnowledgeBase.load(f"{ds}_kb", embedder)
                collect_ds_split(kb, embedder, extractor, ds, split,
                                 _features_path(run_dir, ds, split),
                                 clean=split.startswith("clean"),
                                 graph_mode=args.graph_mode)

    if args.stage in ("train", "all"):
        stage_train(args, run_dir)

    if args.stage in ("eval", "all"):
        from defense.risk import LearnedRisk, RuleRisk
        risk_fn = None
        if any(m.startswith("G") for m in args.methods.split(",")):
            risk_fn = (RuleRisk() if args.risk_kind == "rule"
                       else LearnedRisk.load(str(run_dir / "risk_learner.pkl")))
        stage_eval(args, run_dir, datasets, args.split, llm, embedder, risk_fn)


if __name__ == "__main__":
    main()
