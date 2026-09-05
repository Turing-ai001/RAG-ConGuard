# -*- coding: utf-8 -*-
"""fp/exact_run.py — Phase 3：exact CCI vs surrogate 的相关性（冻结协议 §7/§10）。

    python -m fp.exact_run --split val --run-dir results/p1 [--H 16] [--topk 3000]

对 split 的每个 query：
    1) full 上下文生成 H-token 探针（T=0）
    2) 对每个 signed 联盟：full / minus-C 两次 teacher-forced forward → JSD（exact）
    3) surrogate：trust-weight 答案支持 影响/控制（target-free 候选答案）
输出 results/<run>/exact_cci_{split}.jsonl（每联盟一行）
    + exact_cci_{split}.json：按 (数据集, 联盟规模, 投毒与否, H) 分组的
      Spearman(control【控制合成分】, exact) 与 Spearman(impact, exact)、样本数。
附加：记录每 query 的探针文本与联盟清单（可人工核查）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.stats import spearmanr

from fp.exact_cci import run_exact_cci
from fp.runner import _features_path, load_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--datasets", default="nq,hotpotqa,msmarco")
    ap.add_argument("--run-dir", default="results/p1")
    ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    from llm import get_backend
    from retrieval import Embedder

    embedder = Embedder()   # noqa: F841（确保环境同款）
    llm = get_backend()
    datasets = [d.strip() for d in args.datasets.split(",")]

    rows = []
    t0 = time.perf_counter()
    for ds in datasets:
        for rec in load_features(_features_path(run_dir, ds, args.split)):
            out = run_exact_cci(rec, llm, H=args.H)
            for c in out["coalitions"]:
                rows.append({**c, "dataset": ds, "query_id": rec["query_id"],
                             "question": rec["question"], "probe": out["probe"],
                             "H": args.H})
            print(f"[exact] {ds}/{rec['query_id']} -> {len(out['coalitions'])} "
                  f"coalitions ({time.perf_counter()-t0:.0f}s)", flush=True)

    jsonl = run_dir / f"exact_cci_{args.split}_H{args.H}.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    score = lambda x: x.get("surrogate_control", 0.0)       # noqa: E731
    groups = {}
    for r in rows:
        key = (r["dataset"], r["size"], r["poisoned_ratio"] >= 0.8)
        groups.setdefault(key, []).append(r)
    result = {"H": args.H, "n_coalitions": len(rows)}
    for key, grp in groups.items():
        x = [score(r) for r in grp]
        y = [r["exact_cci"] for r in grp]
        xi = [r.get("surrogate_impact", 0.0) for r in grp]
        if len(grp) >= 3 and len(set(x)) > 1:
            sp_c = spearmanr(x, y)
            sp_i = spearmanr(xi, y) if len(set(xi)) > 1 else (float("nan"), 0.0)
            result[f"{key[0]}|size{key[1]}|poison{key[2]}"] = {
                "n": len(grp),
                "rho_control_vs_exact": round(float(sp_c[0]), 4),
                "rho_impact_vs_exact": round(float(sp_i[0]), 4),
            }
    # 全局
    if len(rows) >= 3:
        sp_c = spearmanr([score(r) for r in rows], [r["exact_cci"] for r in rows])
        sp_i = spearmanr([r.get("surrogate_impact", 0.0) for r in rows],
                         [r["exact_cci"] for r in rows])
        result["overall"] = {
            "rho_control_vs_exact": round(float(sp_c[0]), 4),
            "rho_impact_vs_exact": round(float(sp_i[0]), 4),
        }
    out_json = run_dir / f"exact_cci_{args.split}_H{args.H}.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("[exact] saved", out_json)
    print(json.dumps(result, ensure_ascii=False, indent=2)[:1800])


if __name__ == "__main__":
    main()
