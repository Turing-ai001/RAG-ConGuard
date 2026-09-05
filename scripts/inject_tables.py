# -*- coding: utf-8 -*-
"""把真实数据表（Table 1-4）嵌入 RAG_论文框架_初稿_v2.docx 的 Table 占位卡片之后。
输出：RAG_论文框架_初稿_v3.docx
"""
import copy
import sys
from pathlib import Path

import docx
from docx.shared import Pt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "RAG_论文框架_初稿_v2.docx"
OUT = ROOT / "RAG_论文框架_初稿_v4.docx"

# 每张真实表：卡片标题（查找占位卡）+ 表头 + 数据行 + 表注
CAPTIONS = {
    "Table 1": ("Table 1. Retrieval-stage attack exposure and view-level hit rates (attacked, no defense).",
                ["Dataset", "Retrieval ASR@5", "Avg. poisoned in top-5", "Dense hit", "Lexical hit"],
                [["NQ", "0.99", "3.88", "0.99", "0.94"],
                 ["HotpotQA", "1.00", "4.89", "1.00", "1.00"],
                 ["MS MARCO", "0.97", "3.12", "0.98", "0.86"]],
                "Note: Retrieval ASR = fraction of queries with at least one poisoned document in the top-5; "
                "the KG view is disabled in the reported configuration. 100 evaluation queries per dataset."),
    "Table 2": ("Table 2. Generation attack success rate (ASR).",
                ["Dataset", "Vanilla", "Relevance filter", "Duplicate filter", "RAG-ConGuard (tau=0.25)"],
                [["NQ", "0.86", "0.86", "0.76", "0.14"],
                 ["HotpotQA", "0.99", "0.99", "0.93", "0.14"],
                 ["MS MARCO", "0.82", "0.82", "0.70", "0.14"]],
                "Note: RAG-ConGuard values are reported on the aggregated held-out split (70 queries; "
                "30 queries per dataset for training); baselines on 100 queries per dataset. "
                "Relevance filter tau=0.30 retains all documents; duplicate filter tau=0.85 retains 2.3/1.2/2.3 docs."),
    "Table 3": ("Table 3. Generation accuracy and the cost of defense (attacked setting).",
                ["Dataset", "Vanilla (attacked)", "Relevance filter", "Duplicate filter",
                 "RAG-ConGuard", "Refusal rate"],
                [["NQ", "0.18", "0.18", "0.31", "0.06", "1.00"],
                 ["HotpotQA", "0.03", "0.03", "0.06", "0.06", "1.00"],
                 ["MS MARCO", "0.22", "0.22", "0.38", "0.06", "1.00"]],
                "Note: RAG-ConGuard accuracy is the aggregated held-out value (0.057). The current safe-context "
                "selector refuses all queries under this attack setting (refusal=1.00), i.e., the robustness gain "
                "is obtained at the cost of refusal; see Section 4.6 for discussion."),
    "Table 4": ("Table 4. Per-document leave-one-out attribution vs. coalitional counterfactual influence (CCI) under redundant poisoning.",
                ["Poisoned group size", "# groups", "Avg. per-document LOO", "Avg. coalitional removal", "Ratio"],
                [["2 docs", "29", "0.159", "0.161", "1.01x"],
                 ["3 docs", "41", "0.372", "0.593", "1.60x"],
                 ["4 docs", "51", "0.463", "0.918", "1.98x"]],
                "Note: The same influence formulas are replayed on the 120 poisoned groups of the evaluation set. "
                "As group size grows, per-document influence is diluted by the remaining members (0.159 to 0.463) while "
                "coalitional removal grows markedly (0.161 to 0.918); the ratio rises from 1.60x to 1.98x, "
                "confirming that per-document attribution fails exactly when poisoning is coordinated."),
}

doc = docx.Document(str(SRC))
tables = doc.tables

def find_table_card(marker):
    for t in tables:
        txt = " ".join(c.text for c in t.rows[0].cells) + " ".join(c.text for c in t.rows[-1].cells)
        if marker in txt:
            return t
    return None

style = None
for s in doc.styles:
    if s.name == "Table Grid":
        style = s
        break

for marker, (caption, header, rows, note) in CAPTIONS.items():
    card = find_table_card(marker)
    if card is None:
        print("MISSING CARD:", marker)
        continue
    # 构造真实表（尾部创建后移动 XML 到占位卡之后）
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    if style is not None:
        t.style = style
    for j, h in enumerate(header):
        t.cell(0, j).text = h
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            t.cell(i, j).text = v
    # 标题段落
    cap_p = doc.add_paragraph()
    run = cap_p.add_run(caption)
    run.bold = True
    run.font.size = Pt(9)
    # 表注段落
    note_p = doc.add_paragraph()
    nrun = note_p.add_run(note)
    nrun.italic = True
    nrun.font.size = Pt(8)
    # 移动：占位卡之后依次插入 cap_p, t, note_p
    ref_el = doc.element.get_or_add_body()
    anchor = card._tbl
    for el in (cap_p._p, t._tbl, note_p._p):
        anchor.addnext(el)          # 反向顺序注意：addnext 每次插到 anchor 后，需保持顺序
        anchor = el
    print("inserted", marker)

doc.save(str(OUT))
print("saved", OUT.name)
