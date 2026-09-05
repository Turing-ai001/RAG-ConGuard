# -*- coding: utf-8 -*-
"""临时诊断：train 特征的联盟纯度/规模/MLP 风险分布（只读，不产出协议结果）。"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/RAG")
base = Path("/root/autodl-tmp/RAG/experiments/final_protocol/results/p1")

rows = []
for p in sorted(base.glob("features_*_train.jsonl")):
    for line in p.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        for co in r["coalitions"]:
            rows.append((p.name.split("_")[1], co["size"], co["purity"]))
print("coalitions:", len(rows))
print("size dist:", dict(Counter(x[1] for x in rows)))
pur = [x[2] for x in rows]
print("purity>=0.8 ratio:", round(sum(1 for x in pur if x >= 0.8) / len(pur), 3),
      " purity mean:", round(sum(pur) / len(pur), 3))

for ds in ("nq", "hotpotqa", "msmarco"):
    qs = [json.loads(l) for l in (base / f"features_{ds}_train.jsonl")
          .read_text(encoding="utf-8").split("\n") if l.strip()]
    nc = [len(q["coalitions"]) for q in qs]
    qdoc = [sum(1 for d in q["docs"] if ":poisoned:" in d["doc_id"]) for q in qs]
    print(ds, "coalitions/query:", dict(Counter(nc)),
          "| poisoned-docs in top5:", dict(Counter(qdoc)))

sys.path.insert(0, "/root/autodl-tmp/RAG/experiments/final_protocol")
from fp.runner import _asr_pool, load_features
from defense.risk import LearnedRisk, feature_matrix

feats = []
for ds2 in ("nq", "hotpotqa", "msmarco"):
    feats += load_features(base / f"features_{ds2}_train.jsonl")
pool, labels = _asr_pool(feats)
risk = LearnedRisk(random_state=20260825)
risk.fit(pool, labels)
sc = risk.predict_proba(feature_matrix(pool))
pos = [s for s, y in zip(sc, labels) if y == 1]
neg = [s for s, y in zip(sc, labels) if y == 0]
print("train pos:", len(pos), "neg:", len(neg))
print("neg risk mean/max:", round(sum(neg) / len(neg), 3), round(max(neg), 3))
print("pos risk mean/min:", round(sum(pos) / len(pos), 3), round(min(pos), 3))
print("neg >=0.25:", round(sum(1 for s in neg if s >= 0.25) / len(neg), 3))
print("pos >=0.25:", round(sum(1 for s in pos if s >= 0.25) / len(pos), 3))
