# -*- coding: utf-8 -*-
"""fp/tune_baselines.py — Phase 4 基线在 val 上的参数扫描（B4 无参、B6 τ_loo、B5 无参）。

    python -m fp.tune_baselines --run-dir results/p1

对每个候选在投毒 val（60q）上全生成评估（ASR/acc/refusal），写入
results/<run>/tune_baselines.json。B1/B2 的 τ_rel/τ_dup 同样在此扫描
（B1 网格 0.1-0.5；B2 网格 0.6-0.9）。G-方法已由 fp.tune 处理。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fp.io import main_state, metrics_from_predictions
from fp.runner import METHODS, _features_path, load_features, run_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/p1")
    ap.add_argument("--datasets", default="nq,hotpotqa,msmarco")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    datasets = [d.strip() for d in args.datasets.split(",")]

    from llm import get_backend
    from retrieval import Embedder
    embedder, llm = Embedder(), get_backend()

    feats = []
    for ds in datasets:
        feats += load_features(_features_path(run_dir, ds, "val"))

    candidates = {
        "B4": [{}],
        "B6": [{"tau_loo": t} for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)],
        "B5": [{}],
        "B1": [{"tau_rel": t} for t in (0.1, 0.2, 0.3, 0.4, 0.5)],
        "B2": [{"tau_dup": t} for t in (0.6, 0.7, 0.8, 0.85, 0.9)],
    }
    rows = []
    for mid, cfgs in candidates.items():
        for cfg in cfgs:
            mtype, orig = METHODS[mid]
            METHODS[mid] = (mtype, {**orig, **cfg})
            t0 = time.perf_counter()
            preds = []
            for rec in feats:
                out = run_method(mid, rec, llm, embedder)
                state = main_state(out["answer"] if not out["refused"]
                                   else "I don't know",
                                   rec.get("gold") or "",
                                   rec.get("attack_target") or "")
                preds.append({"state": state, "refused": bool(out["refused"]),
                              "removed_n": len(out["removed"])})
            m = metrics_from_predictions(preds)
            rows.append({"method": mid, "cfg": cfg, "asr": m["asr"],
                         "accuracy": m["accuracy"], "refusal": m["refusal"],
                         "n": m["n"],
                         "avg_removed": round(sum(p["removed_n"]
                                                  for p in preds) / len(preds), 3),
                         "ok_refusal": m["refusal"] <= 0.10})
            print(f"[tuneB] {mid} {cfg} -> asr={m['asr']:.3f} "
                  f"acc={m['accuracy']:.3f} refusal={m['refusal']:.3f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    (run_dir / "tune_baselines.json").write_text(
        json.dumps({"scan": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("saved tune_baselines.json")


if __name__ == "__main__":
    main()
