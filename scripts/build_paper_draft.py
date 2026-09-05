# -*- coding: utf-8 -*-
"""把论文英文初稿每卡内容注入 RAG_论文模仿框架.docx 的 ✏️ 写作区。
输出：RAG_论文框架_初稿.docx（模板+全部写作区填写）。"""
import json
from pathlib import Path
import docx

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "RAG_论文模仿框架.docx"
OUT = ROOT / "RAG_论文框架_初稿_v2.docx"

cards = json.loads((ROOT / "paper_draft_cards_full.json").read_text(encoding="utf-8"))

# 文档中写作区的出现顺序（与框架卡片顺序一致）
CARD_ORDER = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
              "I1", "I2", "I3", "I4", "I5", "I6", "I7",
              "RW1a", "RW1b", "RW2a", "RW2b",
              "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10",
              "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10",
              "E11", "E12",
              "CLHAT", "CL1", "CL2", "CL3", "CL4"]

doc = docx.Document(str(SRC))
idx = 0
replaced = 0
for p in doc.paragraphs:
    if p.text.strip().startswith("✏️") or "Write your content here" in p.text:
        key = CARD_ORDER[idx]
        text = cards[key]
        # 保留原段落文本风格：清空 writing 占位、写入卡片内容
        p.text = text
        replaced += 1
        idx += 1
        if idx >= len(CARD_ORDER):
            break

doc.save(str(OUT))
print(f"replaced {replaced} writing sections -> {OUT.name}")
assert replaced == len(CARD_ORDER), f"expected {len(CARD_ORDER)}, got {replaced}"
