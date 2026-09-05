# -*- coding: utf-8 -*-
"""合并三批卡片文本为 paper_draft_cards_full.json（批次内容已固化在本文件）。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

batch1 = json.loads((ROOT / "scripts" / "cards_batch1.json").read_text(encoding="utf-8"))
batch2 = json.loads((ROOT / "scripts" / "cards_batch2.json").read_text(encoding="utf-8"))
batch3 = json.loads((ROOT / "paper_draft_cards.json").read_text(encoding="utf-8"))
full = {**batch1, **batch2, **batch3}
(ROOT / "paper_draft_cards_full.json").write_text(
    json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
print("full cards:", len(full))
