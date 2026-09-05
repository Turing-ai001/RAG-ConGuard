# -*- coding: utf-8 -*-
"""fp/tune.py — val 上的 operating point 调参（冻结协议 §8，只许用 val）。

对 S0(τ)/S1(τ)/S2(λ) 在投毒 val 上扫描（每个候选都必须真实生成，
因为 ASR/refusal 是生成结果）。选择规则（协议 §8）：
    1) Refusal ≤ max_refusal（0.10）          ← 优先约束
    2) clean accuracy drop ≤ max_clean_acc_drop（仅对 finally chosen 复算 clean）
    3) 满足约束的点中取 ASR 最小；若无满足点 → refusal 最小者，并记 constraint_ok=False

输出 results/<run>/tune.json：{scan, chosen, clean_B0_ref}
之后跑 eval 时以 chosen 参数发布（test 只跑一次）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root

from config import RAG_PROMPT

from fp.io import main_state, metrics_from_predictions
from fp.runner import (_features_path, load_features, run_method)
from fp.selectors import SELECTORS


def _ctx(docs):
    return "\n".join(x["text"] for x in docs)


def run_conguard_cfg(rec: dict, selector: str, params: dict, llm, risk_fn):
    sel = SELECTORS[selector](
        rec["docs"], rec["claims"], rec["coalitions"], risk_fn,
        **{k: v for k, v in params.items()})
    if sel["refused"]:
        return {"answer": "I don't know", "refused": True,
                "removed_n": len(sel["removed_docs"]),
                "kept_n": len(sel["kept_docs"])}
    ans = llm.complete(RAG_PROMPT.replace("[question]", rec["question"])
                       .replace("[context]", _ctx(sel["kept_docs"])))
    return {"answer": ans, "refused": False,
            "removed_n": len(sel["removed_docs"]),
            "kept_n": len(sel["kept_docs"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/p1")
    ap.add_argument("--datasets", default="nq,hotpotqa,msmarco")
    ap.add_argument("--max-refusal", type=float, default=0.10)
    ap.add_argument("--max-clean-drop", type=float, default=0.05)
    ap.add_argument("--s0-taus", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    ap.add_argument("--s1-taus", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    ap.add_argument("--s2-lams", default="0.2,0.35,0.5,0.6,0.75,1.0")
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    datasets = [d.strip() for d in args.datasets.split(",")]

    from defense.risk import LearnedRisk
    from llm import get_backend
    from retrieval import Embedder

    embedder = Embedder()
    llm = get_backend()
    risk_fn = LearnedRisk.load(str(run_dir / "risk_learner.pkl"))

    # 参考：clean_val 上 B0 的干净准确率（只报告，不用于候选过滤）
    import numpy as np
    clean_preds = []
    for ds in datasets:
        feats = load_features(_features_path(run_dir, ds, "clean_val"))
        for rec in feats:
            out = run_method("B0", rec, llm, embedder, risk_fn)
            clean_preds.append({"dataset": ds, "query_id": rec["query_id"],
                                "method": "clean_B0",
                                "state": main_state(out["answer"], "", ""),
                                "refused": False, "removed_n": 0})
    b0_clean = metrics_from_predictions(clean_preds)

    params_list = {
        "s0": [{"tau": float(t)} for t in args.s0_taus.split(",")],
        "s1": [{"tau": float(t)} for t in args.s1_taus.split(",")],
        "s2": [{"lam": float(l)} for l in args.s2_lams.split(",")],
    }
    selectors = [s for s in ("s0", "s1", "s2") if params_list[s]]

    feats = {}
    for ds in datasets:
        feats[ds] = load_features(_features_path(run_dir, ds, "val"))
    n_val = sum(len(v) for v in feats.values())

    rows = []
    for sel in selectors:
        for params in params_list[sel]:
            t0 = time.perf_counter()
            preds = []
            for ds in datasets:
                for rec in feats[ds]:
                    out = run_conguard_cfg(rec, sel, params, llm, risk_fn)
                    state = main_state(out["answer"], rec.get("gold") or "",
                                       rec.get("attack_target") or "")
                    preds.append({"dataset": ds, "query_id": rec["query_id"],
                                  "method": f"tune_{sel}", "state": state,
                                  "refused": bool(out["refused"]),
                                  "removed_n": out["removed_n"],
                                  "kept_n": out["kept_n"]})
            m = metrics_from_predictions(preds)
            rows.append({
                "selector": sel, "params": params,
                "asr": m["asr"], "accuracy": m["accuracy"],
                "refusal": m["refusal"], "n": m["n"],
                "avg_removed": round(sum(p["removed_n"] for p in preds) /
                                     len(preds), 3),
                "ok_refusal": m["refusal"] <= args.max_refusal,
            })
            print(f"[tune] {sel} {params} -> asr={m['asr']:.3f} "
                  f"refusal={m['refusal']:.3f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    feasible = [r for r in rows if r["ok_refusal"]]
    pool = feasible or rows
    best = min(pool, key=lambda r: r["asr"])
    chosen = {"selector": best["selector"], "params": best["params"],
              "asr": best["asr"], "refusal": best["refusal"],
              "constraint_ok": bool(feasible)}
    (run_dir / "tune.json").write_text(json.dumps(
        {"scan": rows, "chosen": chosen,
         "clean_B0_ref": b0_clean}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("[tune] CHOSEN:", json.dumps(chosen, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
