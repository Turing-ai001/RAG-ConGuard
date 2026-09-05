# -*- coding: utf-8 -*-
"""fp/data.py — split/数据装载（冻结协议 §1-§2，只读）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
from config import POISONED_DIR, RAW_DIR
from retrieval.kb import load_poisoned_docs

FP_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = FP_ROOT / "splits"


def _read_ids(kind: str) -> dict:
    p = SPLITS_DIR / f"{kind}_ids.json"
    if not p.exists():
        raise FileNotFoundError(f"split not found: {p} —— 先跑 scripts/make_splits.py")
    return json.loads(p.read_text(encoding="utf-8"))


def split_ids(dataset: str, kind: str) -> list[str]:
    """kind ∈ train | val | test | clean_val | clean_test。"""
    return _read_ids(kind)[dataset]


def iter_split_queries(datasets: list[str], kind: str):
    """按 (dataset, qid, item) 顺序遍历冻结 split 的投毒查询。"""
    for ds in datasets:
        data = load_poisoned_docs(ds)
        for qid in split_ids(ds, kind):
            yield ds, qid, data[qid]


def clean_query_file(dataset: str) -> str:
    return RAW_DIR / f"{dataset}_queries.jsonl"


def clean_golds(dataset: str) -> dict:
    """BEIR qas（若已 fetch）：id → gold answer 列表。NQ/HotpotQA 有，MS MARCO 无。"""
    p = RAW_DIR / f"{dataset}_qas.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rec = json.loads(line)
            out[rec["_id"]] = rec.get("answers", [])
    return out


def clean_queries(dataset: str, kind: str) -> list[dict]:
    """干净查询条目。返回 [{id, question, gold}]（question = queries.jsonl text；
    gold = qas answers 第一个或 None——MS MARCO 无 qas 则 None → 不出准确率）。"""
    ids = set(split_ids(dataset, kind))
    golds = clean_golds(dataset)
    out = []
    for line in (clean_query_file(dataset)).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["_id"] in ids:
            ans = golds.get(rec["_id"], [])
            out.append({"id": rec["_id"], "question": rec["text"],
                        "gold": (ans[0] if ans else None)})
    return out
