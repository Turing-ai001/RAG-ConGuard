# -*- coding: utf-8 -*-
"""生成冻结 split（FINAL_PROTOCOL §2，只运行一次，产物进 splits/）。

    python scripts/make_splits.py

输出（每 JSON = {"nq": [ids], "hotpotqa": [...], "msmarco": [...]}）：
    splits/train_ids.json / val_ids.json / test_ids.json   （投毒查询 60/20/20，分层）
    splits/clean_val_ids.json / clean_test_ids.json        （干净查询 100/300，分层，排除投毒 id）

确定性：sorted(ids) → random.Random(20260825).sample(id_count) → 切片。
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from config import POISONED_DIR, RAW_DIR

SEED = 20260825
DATASETS = ["nq", "hotpotqa", "msmarco"]
SPLIT = {"train": 60, "val": 20, "test": 20}
CLEAN = {"trn": 200, "val": 100, "test": 300}
OUT = Path(__file__).resolve().parent.parent / "splits"


def poisoned_ids(ds: str) -> list[str]:
    data = json.loads((POISONED_DIR / f"{ds}.json").read_text(encoding="utf-8"))
    return sorted(data.keys())


def clean_ids(ds: str) -> list[str]:
    poisoned = set(poisoned_ids(ds))
    ids = []
    for line in (RAW_DIR / f"{ds}_queries.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rec = json.loads(line)
            if rec["_id"] not in poisoned:
                ids.append(rec["_id"])
    return ids


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {"train": {}, "val": {}, "test": {}, "clean_trn": {},
              "clean_val": {}, "clean_test": {}}
    for ds in DATASETS:
        ids = poisoned_ids(ds)
        assert len(ids) == 100, f"{ds}: poisoned queries={len(ids)} != 100"
        order = random.Random(SEED).sample(ids, len(ids))
        splits["train"][ds] = order[: SPLIT["train"]]
        splits["val"][ds] = order[SPLIT["train"]: SPLIT["train"] + SPLIT["val"]]
        splits["test"][ds] = order[SPLIT["train"] + SPLIT["val"]:]

        cids = clean_ids(ds)
        tot_clean = CLEAN["trn"] + CLEAN["val"] + CLEAN["test"]
        assert len(cids) >= tot_clean, f"{ds}: clean={len(cids)} too few"
        corder = random.Random(SEED + 1).sample(cids, len(cids))
        splits["clean_trn"][ds] = corder[: CLEAN["trn"]]
        splits["clean_val"][ds] = corder[CLEAN["trn"]:
                                         CLEAN["trn"] + CLEAN["val"]]
        splits["clean_test"][ds] = corder[CLEAN["trn"] + CLEAN["val"]: tot_clean]
        print(f"{ds}: poison={len(ids)} "
              f"({SPLIT['train']}/{SPLIT['val']}/{SPLIT['test']}) "
              f"clean_sample={len(corder)} "
              f"(trn {CLEAN['trn']}, val {CLEAN['val']}, test {CLEAN['test']})")

    for kind, data in splits.items():
        path = OUT / f"{kind}_ids.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"-> {path}")
    # 交叉检查：train/val/test 无交集
    for ds in DATASETS:
        s = [set(splits[k][ds]) for k in ("train", "val", "test")]
        assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2]), ds
        cleanup = set(splits["clean_val"][ds]) & set(splits["clean_test"][ds])
        assert not cleanup, f"clean overlap {ds}"
    print("splits disjointness OK")


if __name__ == "__main__":
    main()
