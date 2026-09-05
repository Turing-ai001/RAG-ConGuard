"""P2 评估：联盟检测（P/R/F1）+ 反事实影响判别力（AUC）+ 防御模拟曲线。

流程（每 query）：
    query → KB 多视角检索（TOP_K）→ claims（meta 标注目标投毒）
    → RelationExtractor（NLI 主干 + LLM 低置信兜底）→ 有符号 ClaimGraph
    → 联盟发现（正边连通分量）→ 联盟特征（coalition.analyze）
    → 信任传播/答案支持 + 反事实影响（coalition.trust）

指标：
    1) 联盟检测   纯度>=P2_GT_PURITY 且 size>=P2_MIN_COALITION_SIZE 的联盟
                  = 检出的投毒联盟（节点集级 P/R/F1，GT=目标投毒 claims）
    2) 判别力     联盟 impact / control 区分"投毒 vs 非投毒联盟" 的 AUC
    3) 防御模拟   按 control 从高到低删联盟（tau 阈值）→ 投毒作答率曲线

用法：
    python scripts/eval_p2.py --dataset all --num_queries 100
    python scripts/eval_p2.py --dataset nq --num_queries 10 --no-llm --synthetic-normal 2
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

from coalition.analyze import coalition_features
from coalition.trust import answer_support, counterfactual_impacts, trust_weights
from config import (INDEX_DIR, OUTPUT_DIR, P2_GT_PURITY, P2_IMPACT_TAUS, TOP_K)
from graph import ClaimGraph, signed_semantic_graph
from llm import get_backend
from relation import LLMJudge, NLIScorer, RelationExtractor
from retrieval import Embedder, KnowledgeBase
from retrieval.kb import load_poisoned_docs


def synthetic_normal_claims(item: dict, n: int) -> list[dict]:
    """从正确答案构造 n 条'正常' claim（mini KB 无正常语料时制造矛盾对）。"""
    q = item["question"].rstrip("?").lower()
    ans = item["correct answer"]
    template = (
        "According to the official records, the answer to \"{q}\" is {ans}, "
        "and this is confirmed by multiple independent sources."
    )
    return [
        {"claim_id": f"normal-{i}", "text": template.format(q=q[:60], ans=ans)[:220],
         "meta": {"poisoned": False}}
        for i in range(n)
    ]


def doc_to_claim_id(doc_id: str) -> str:
    parts = doc_id.split(":")
    return parts[-2][:8]


def get_kb(embedder, dataset: str) -> KnowledgeBase:
    """优先全量 KB（{ds}_kb），否则 mini KB（仅投毒文档）。"""
    kb_root = INDEX_DIR / f"{dataset}_kb"
    if kb_root.exists():
        return KnowledgeBase.load(f"{dataset}_kb", embedder)
    print(f"[eval_p2] !! {dataset}_kb 不存在，回退 mini KB（仅投毒文档，"
          f"需要 --synthetic-normal>0 才有正常 claims）")
    return __import__("retrieval.kb", fromlist=["build_mini_kb"]).build_mini_kb(embedder, dataset)


def run_one(kb, embedder, q: str, item: dict, extractor, synth_normal: int,
             graph_mode: str = "signed") -> dict:
    docs = kb.search(q, top_k=TOP_K)
    claims = []
    for d in docs:
        is_target = (":poisoned:" in d["doc_id"]
                     and d.get("meta", {}).get("target_query_id") == item.get("id"))
        cid = f"{doc_to_claim_id(d['doc_id'])}-{len(claims)}"
        claims.append({"claim_id": cid, "text": d["text"][:220],
                       "meta": {"poisoned": is_target}})
    claims += synthetic_normal_claims(item, synth_normal)

    edges = extractor.extract_all(claims)
    g = (signed_semantic_graph(claims, edges, embedder, threshold=0.85)
         if graph_mode == "signed_semantic" else ClaimGraph(claims, edges))
    coalitions = g.coalition_nodes("signed")
    feats = coalition_features(g, coalitions)
    fmap = {tuple(f["claims"]): f for f in feats}

    # 候选答案：poisoned.json 的 correct / incorrect answer（BGE 余弦相似度）
    answer_ids = ["correct", "poisoned"]
    a_texts = [item["correct answer"], item["incorrect answer"]]
    c_embs = embedder.encode([c["text"] for c in claims])
    a_embs = embedder.encode(a_texts)
    c_embs = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-9)
    a_embs = a_embs / (np.linalg.norm(a_embs, axis=1, keepdims=True) + 1e-9)
    sims = c_embs @ a_embs.T                     # (n_claims, 2)

    w0 = trust_weights(g)
    s0 = answer_support(claims, w0, sims, answer_ids)
    impacts = counterfactual_impacts(g, coalitions, claims, sims, answer_ids)

    coal_records = []
    for imp in impacts:
        feat = fmap.get(tuple(imp["claims"]), {})
        corr = feat.get("external_corroboration", 0.0)
        purity = _coalition_purity(imp["claims"], claims)
        coal_records.append({
            "claims": imp["claims"],
            "size": imp["size"],
            "purity": purity,
            "impact": imp["impact"],
            "control": round(imp["impact"] * (1 - min(corr, 1.0)), 4),
            "cohesion": feat.get("cohesion", 0.0),
            "external_corroboration": corr,
            "external_refute": feat.get("external_refute", 0.0),
        })
    coal_records.sort(key=lambda c: -c["control"])

    return {"query": q, "claims": claims, "edges": edges,
            "sims": sims, "answer_ids": answer_ids,
            "s0": s0, "argmax_before": max(s0, key=s0.get),
            "coalitions": coal_records,
            "gt_claims": [c["claim_id"] for c in claims if c["meta"].get("poisoned")]}


def _coalition_purity(node_ids: list[str], claims: list[dict]) -> float:
    poisoned = {c["claim_id"] for c in claims if c["meta"].get("poisoned")}
    n = len(node_ids)
    if n == 0:
        return 0.0
    r = sum(1 for cid in node_ids if cid in poisoned) / n
    return round(max(r, 1 - r), 3)


def simulate_defense(r: dict, tau: float, min_purity: float = 0.6) -> str:
    """删除 control>=tau 且 purity>=min_purity 的联盟后，图上的投毒作答情况。

    返回 argmax 答案 id；图删空（拒答）→ "none"（视为防御成功）。
    """
    claims = r["claims"]; edges = r["edges"]; sims = r["sims"]
    drop = set()
    for c in r["coalitions"]:
        if c["control"] >= tau and c["purity"] >= min_purity:
            drop.update(c["claims"])
    keep = [c for c in claims if c["claim_id"] not in drop]
    keep_edges = [e for e in edges if e.src not in drop and e.dst not in drop]
    if not keep:
        return "none"
    g2 = ClaimGraph(keep, keep_edges)
    w2 = trust_weights(g2)
    idx = {c["claim_id"]: i for i, c in enumerate(claims)}
    sub_sims = sims[[idx[c["claim_id"]] for c in keep]]
    s2 = answer_support(keep, w2, sub_sims, r["answer_ids"])
    return max(s2, key=s2.get)


def _jsonable(obj):
    """JSON 兜底序列化：numpy 数组/标量 → python 原生；Relation → to_dict()。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, "to_dict"):            # Relation（__slots__ 类）
        return _jsonable(obj.to_dict())
    raise TypeError(f"not JSON-serializable: {type(obj)}")


def _auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney U 的 ROC AUC（无 scikit 依赖）。"""
    pos = sorted(s for s, y in zip(scores, labels) if y == 1)
    neg = sorted(s for s, y in zip(scores, labels) if y == 0)
    if not pos or not neg:
        return float("nan")
    j = 0
    rank_sum = 0.0
    for s in pos:
        while j < len(neg) and neg[j] < s:
            j += 1
        rank_sum += j
    return rank_sum / (len(pos) * len(neg))


def main(datasets: list[str], num_queries: int, use_llm: bool, synth_normal: int,
         graph_mode: str = "signed"):
    embedder = Embedder()
    nli = NLIScorer()
    judge = LLMJudge(get_backend()) if use_llm else None
    extractor = RelationExtractor(nli, judge)

    coalition_pool = []          # AUC 用的联盟池
    per_query_f1 = []
    baseline = {"poisoned": 0, "ok": 0}
    defense = {t: {"poisoned": 0, "ok": 0} for t in P2_IMPACT_TAUS}
    rows = []

    for ds in datasets:
        kb = get_kb(embedder, ds)
        poisoned = load_poisoned_docs(ds)
        for qid, item in list(poisoned.items())[:num_queries]:
            item = {**item, "id": qid}
            r = run_one(kb, embedder, item["question"], item, extractor, synth_normal,
                        graph_mode=graph_mode)
            rows.append({"dataset": ds, "query_id": qid, **r})
            gt = set(r["gt_claims"])
            coals = r["coalitions"]

            # 1) 联盟检测 P/R/F1（节点集级；GT 为空视为 1.0 无惩罚）
            detected = set()
            for c in coals:
                if c["purity"] >= P2_GT_PURITY and len(c["claims"]) >= 2:
                    detected.update(c["claims"])
            p = len(detected & gt) / len(detected) if detected else 1.0
            rc = len(detected & gt) / len(gt) if gt else 1.0
            f1 = 2 * p * rc / (p + rc) if p + rc > 0 else 0.0
            per_query_f1.append({"p": p, "r": rc, "f1": f1,
                                 "n_gt": len(gt), "n_pred": len(detected)})

            # 2) 联盟池（AUC）：投毒标注沿用 P1 口径 purity>=0.8
            for c in coals:
                coalition_pool.append({**c, "dataset": ds, "query_id": qid,
                                       "poisoned": 1 if c["purity"] >= 0.8 else 0})

            # 3) 防御曲线
            if r["argmax_before"] == "poisoned":
                baseline["poisoned"] += 1
            else:
                baseline["ok"] += 1
            for t in P2_IMPACT_TAUS:
                post = simulate_defense(r, t)
                if post == "poisoned":
                    defense[t]["poisoned"] += 1
                else:
                    defense[t]["ok"] += 1

    # ---------- 汇总 ----------
    print("=" * 72)
    print(f"P2 评估 | datasets={datasets} | queries={num_queries}/数据集 | "
          f"LLM兜底={use_llm} | 合成正常claims={synth_normal} | "
          f"联盟数={len(coalition_pool)}")
    print("=" * 72)

    ps = float(np.mean([x["p"] for x in per_query_f1]))
    rs = float(np.mean([x["r"] for x in per_query_f1]))
    fs = float(np.mean([x["f1"] for x in per_query_f1]))
    print(f"[联盟检测] 平均 P={ps:.3f} R={rs:.3f} F1={fs:.3f} "
          f"(纯度>={P2_GT_PURITY} 且 大小>=2; n={len(per_query_f1)})")

    for key in ("impact", "control"):
        auc = _auc([x[key] for x in coalition_pool],
                   [x["poisoned"] for x in coalition_pool])
        n_pos = sum(1 for x in coalition_pool if x["poisoned"])
        print(f"[{key} 判别投毒联盟 AUC] {auc:.3f} "
              f"({n_pos} 投毒 / {len(coalition_pool) - n_pos} 非投毒)")

    n = baseline["poisoned"] + baseline["ok"] or 1
    print(f"\n[防御曲线] 无防御 投毒作答率 = "
          f"{baseline['poisoned'] / n:.3f}（n={n} queries；'none' 视作防御成功）")
    for t, st in defense.items():
        tot = st["poisoned"] + st["ok"] or 1
        print(f"  tau={t:<5} → 投毒作答率 = {st['poisoned'] / tot:.3f}")

    out = OUTPUT_DIR / "eval_pipeline" / "eval_p2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"datasets": datasets, "num_queries": num_queries,
                   "use_llm": use_llm, "synth_normal": synth_normal,
                   "gt_purity": P2_GT_PURITY, "graph_mode": graph_mode},
        "detection_f1": {"p": ps, "r": rs, "f1": fs},
        "auc": {k: _auc([x[k] for x in coalition_pool],
                        [x["poisoned"] for x in coalition_pool])
                for k in ("impact", "control")},
        "defense_curve": {str(t): defense[t] for t in P2_IMPACT_TAUS},
        "baseline": baseline,
        "rows": rows,
    }, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    print(f"\n[eval_p2] saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", help="all | nq,hotpotqa,msmarco")
    ap.add_argument("--num_queries", type=int, default=10)
    ap.add_argument("--no-llm", action="store_true", help="禁用 LLM 兜底（纯 NLI）")
    ap.add_argument("--synthetic-normal", type=int, default=0,
                    help="混入合成正常 claims（mini KB 时设 2）")
    ap.add_argument("--graph", default="signed", choices=["signed", "signed_semantic"],
                    help="图变体：signed（NLI 符号图）| signed_semantic（+BGE 语义边）")
    args = ap.parse_args()
    ds = ["nq", "hotpotqa", "msmarco"] if args.dataset == "all" \
        else [d.strip() for d in args.dataset.split(",")]
    main(ds, args.num_queries, use_llm=not args.no_llm,
         synth_normal=args.synthetic_normal, graph_mode=args.graph)
