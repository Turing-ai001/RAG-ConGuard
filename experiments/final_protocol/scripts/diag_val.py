# -*- coding: utf-8 -*-
"""Phase 2 诊断（读 p1 val 特征，CPU 为主 + 少量生成）：
  1) val 无防御 B0 ASR（60 查询 3 数据集，真实生成）
  2) val 池 MLP / rule AUC
  3) 联盟结构：投毒 vs 良性在联盟中的交错（合并度）分布
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/RAG")
sys.path.insert(0, "/root/autodl-tmp/RAG/experiments/final_protocol")

from defense.risk import LearnedRisk, RuleRisk, feature_matrix
from fp.io import auc, main_state, metrics_from_predictions
from fp.runner import load_features, _features_path, run_method
from llm import get_backend
from retrieval import Embedder

base = Path("/root/autodl-tmp/RAG/experiments/final_protocol/results/p1")
datasets = ["nq", "hotpotqa", "msmarco"]

feats = {ds: load_features(_features_path(base, ds, "val")) for ds in datasets}

# ---- 1) val B0 ----
embedder = Embedder()
llm = get_backend()
preds = []
for ds in datasets:
    for rec in feats[ds]:
        out = run_method("B0", rec, llm, embedder)
        ans = out["answer"] if not out["refused"] else "I don't know"
        preds.append({"dataset": ds, "query_id": rec["query_id"],
                      "state": main_state(ans, rec.get("gold") or "",
                                          rec.get("attack_target") or ""),
                      "refused": bool(out["refused"])})
m = metrics_from_predictions(preds)
print("[val B0]", json.dumps(m, ensure_ascii=False)[:250], flush=True)

# ---- 2) AUC ----
pool, labels = [], []
for ds in datasets:
    for rec in feats[ds]:
        for c in rec["coalitions"]:
            pool.append({**c, "dataset": ds, "query_id": rec["query_id"]})
            labels.append(1 if c.get("poisoned_ratio", 0.0) >= 0.8 else 0)
risk = LearnedRisk.load(str(base / "risk_learner.pkl"))
sc = risk.predict_proba(feature_matrix(pool)).tolist()
print("[val AUC] learned:", round(auc(sc, labels), 4),
      "| rule(control):", round(auc([r["control"] for r in pool], labels), 4),
      "| n pool:", len(pool), flush=True)
# 阈值扫描洞察：不同 τ 的逃逸（投毒联盟未被标记）
for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
    miss = sum(1 for s, y in zip(sc, labels) if y == 1 and s < t)
    fp = sum(1 for s, y in zip(sc, labels) if y == 0 and s >= t)
    print(f"[val tau={t}] poisoned-below(miss)={miss} benign-above(fp)={fp}", flush=True)

# ---- 3) 结构诊断 ----
for ds in datasets:
    sizes = Counter()
    merged_benign_in_poisoned = 0
    benign_alone = 0
    for rec in feats[ds]:
        for c in rec["coalitions"]:
            sizes[c["size"]] += 1
            po = c["poisoned_ratio"]
            if 0.2 < po < 0.8:
                merged_benign_in_poisoned += 1
            if po == 0.0:
                benign_alone += 1
    print(f"[val {ds}] coalition-size dist: {dict(sizes)} | "
          f"mixed coalitions={merged_benign_in_poisoned} | all-benign={benign_alone}",
          flush=True)
