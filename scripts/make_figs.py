# -*- coding: utf-8 -*-
"""RAG-ConGuard 论文 Figure 1-4 绘制（Fig.5/6 图号在正文中已改为 Fig.3/4）。

数据来源（真实实验输出，勿手改数字）：
    - Fig.3 (ablation):  output/eval_pipeline/ablation_signed.json, eval_p3.json
    - Fig.4 (tau sweep): output/eval_pipeline/eval_p3.json
ablation 的 summary 键存的是 90 次查询的 SUM，口径按原报告方式 /n(=90)
（得 0.972/0.891/0.850，0.124/0.455/0.000）。

输出：figures/fig{1,2,3,4}_*.{pdf,png} + figures/manifest.json
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
DATA = ROOT / "output" / "eval_pipeline"

# Okabe-Ito（skill assets/color_palettes.py，白底 3:1 通过者）
BLUE = "#0072B2"     # 本文方法 / 良性
VERM = "#D55E00"     # 投毒 / 攻击
PURP = "#CC79A7"     # 无符号变体
GRAY = "#BBBBBB"     # 中性 / 相似度基线
GRAYT = "#555555"
BLACK = "#000000"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.linewidth": 0.6,
    "savefig.facecolor": "white",
})


def sbox(ax, x, y, w, h, text, fc="white", ec=BLACK, lw=0.9, fs=6.5,
         hatch=None, pad=0.25):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}", fc=fc,
                       ec=ec, lw=lw, mutation_aspect=1)
    if hatch:
        b.set_hatch(hatch)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.3)
    return b


def arrow(ax, x1, y1, x2, y2, color=BLACK, lw=0.9, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", linewidth=lw,
                        color=color, linestyle=ls, mutation_scale=8)
    ax.add_patch(a)
    return a


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", facecolor="white")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=400, facecolor="white")
    plt.close(fig)
    print("saved", name, ".pdf/.png")


# ---------------------------------------------------------------- Fig. 1 情景图
def fig1():
    fig, ax = plt.subplots(figsize=(152 / 25.4, 70 / 25.4),
                           layout="constrained")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    sbox(ax, 1, 62, 13, 22, "Query q", fs=7)
    ax.text(7.5, 59, "(attacker's\ntarget question)", ha="center", fontsize=5.8,
            color=GRAYT)
    arrow(ax, 14.5, 73, 19.5, 73)
    sbox(ax, 19.5, 64, 15, 22, "Multi-view\nretrieval\n(BGE + BM25)", fs=6.3)
    arrow(ax, 35, 73, 40.5, 73)

    # 证据集合面板
    sbox(ax, 40.5, 40, 27, 58, "Retrieved evidence set D (k = 5)", fs=6.6)
    for i, lid in enumerate(["P1", "P2", "P3"]):
        sbox(ax, 43.5, 83 - i * 11, 7.5, 8.5, lid, fc=VERM, fs=6.6, pad=0.12)
    ax.text(52.5, 82, "near-duplicate poisoned passages\nasserting X",
            fontsize=5.6, color=VERM, va="center")
    for i, lid in enumerate(["B1", "B2"]):
        sbox(ax, 43.5, 52 - i * 11, 7.5, 8.5, lid, fc=BLUE, fs=6.6, pad=0.12)
    ax.text(52.5, 51, "benign documents\n(topically relevant)", fontsize=5.6,
            color=BLUE, va="center")
    arrow(ax, 67.5, 73, 72.5, 73)

    # 图面板（简绘）
    sbox(ax, 72.5, 40, 26.5, 58, "Signed evidence graph", fs=6.6)
    npos = {"P1": (80, 80), "P2": (90, 74), "P3": (80, 66),
            "B1": (90, 47), "B2": (80, 51)}
    for lid, (x, y) in npos.items():
        fc = VERM if lid.startswith("P") else BLUE
        lw = 0.9
        sbox(ax, x - 3.4, y - 3.4, 6.8, 6.8, lid, fc=fc, fs=5.9, pad=0.1)
    for a, b in zip(["P1", "P2"], ["P2", "P3"]):
        arrow(ax, *npos[a], *npos[b], color=VERM, lw=1.0)
    ax.text(85, 88.5, "internal support", fontsize=5.4, color=VERM, ha="center")
    ax.text(85, 62, "no external\ncorroboration", fontsize=5.4, color=GRAYT,
            ha="center")

    # 底部机制对比
    sbox(ax, 1, 10, 48, 19,
         "Per-document LOO\nremoving ONE P: the remaining Ps\nkeep the "
         "group in control\nper-doc influence diluted",
         fc="#FFF6F0", ec=VERM, fs=6.0)
    sbox(ax, 51, 10, 48, 19,
         "Coalitional removal (CCI)\nremoving the WHOLE coalition:\nsupport "
         "collapses -- disentangles\nredundant control",
         fc="#F0F6FF", ec=BLUE, fs=6.0)
    arrow(ax, 84, 39.5, 70, 29.5, color=GRAYT)
    arrow(ax, 63, 39.5, 30, 29.5, color=GRAYT)
    save(fig, "fig1_scenario")


# ---------------------------------------------------------------- Fig. 2 框架图
def fig2():
    fig, ax = plt.subplots(figsize=(170 / 25.4, 95 / 25.4),
                           layout="constrained")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    XS = [2, 15.5, 29, 42.5, 56, 69.5, 83]
    Wd, H, Y = 11.5, 20, 72
    labels = [
        "Multi-view\nretrieval\n(BGE + BM25)",
        "Claim extraction\n(220-char\nevidence units)",
        "Signed evidence\ngraph\n(NLI + numeric\nconflict + judge)",
        "Coalition\ndiscovery\n(candidate →\ncounterfactual\nrefinement)",
        "CCI:\ncoalitional\ncounterfactual\ninfluence\nJSD probe / "
        "trust-weight\napproximation",
        "Risk learner\n(shallow MLP\nover 6 structural\n+ influence\nfeatures)",
        "Safe context\nselection\n(remove C with\nrisk >= tau)",
    ]
    for x, lab in zip(XS, labels):
        sbox(ax, x, Y, Wd, H, lab, fs=5.9)
        if x != XS[-1]:
            arrow(ax, x + Wd, Y + H / 2, x + 57.5, Y + H / 2)
    # 阶段标签
    ax.text(15.5, 94.5, "contextual evidence structuring", fontsize=6.2,
            color=BLUE, ha="center", style="italic")
    ax.text(59.5, 94.5, "coalition-level attribution", fontsize=6.2,
            color=VERM, ha="center", style="italic")
    ax.text(88.8, 94.5, "risk-constrained context construction", fontsize=6.2,
            color=PURP, ha="center", style="italic")

    # 第二排：生成与拒答
    arrow(ax, 88.75, 60, 88.75, 52)
    sbox(ax, 30, 40, 18, 9, "Generator\n(Qwen2.5-7B)", fs=6.0)
    arrow(ax, 48.5, 44.5, 53.5, 44.5)
    sbox(ax, 53.5, 40, 22, 9, "Answer (safe evidence)",
         fc="#F0F6FF", ec=BLUE, fs=6.0)
    arrow(ax, 76, 44.5, 79.5, 44.5)
    sbox(ax, 79.5, 40, 19.5, 9, "Refusal: ``I don't know''",
         fc="#FFF6F0", ec=VERM, fs=6.0)
    arrow(ax, 88.75, 52, 39, 52, color=GRAYT)  # 分流线（视觉）

    # 底层：攻防循环
    sbox(ax, 6, 6, 40, 11,
         "Coalition-aware adaptive attack\npreserve assertion, rewrite style "
         "(corroborate / de-coordinate / mixed)", fs=5.6, ec=GRAYT, hatch="//")
    sbox(ax, 54, 6, 40, 11,
         "Hard-evasion pool\n(evasive variants + benign negatives)\n"
         "retrain risk learner", fs=5.6, ec=GRAYT, hatch="//")
    # 从 Safe context selection 下来（虚线）→ 攻击 → 硬例池 → Risk learner
    arrow(ax, 26, 40, 26, 18, color=GRAYT, ls="--")
    arrow(ax, 46.5, 11.5, 53.5, 11.5, color=GRAYT, ls="--")
    arrow(ax, 74, 17, 74, 20.5, color=GRAYT, ls="--")
    ax.text(26, 30, "evade/reject loop", fontsize=5.8, color=GRAYT,
            rotation=90, ha="center", style="italic")
    ax.text(74, 28.5, "retrain", fontsize=5.8, color=GRAYT, rotation=90,
            ha="center", style="italic")
    save(fig, "fig2_framework")


# ---------------------------------------------------------------- Fig. 3 消融
def fig3():
    ab = json.loads((DATA / "ablation_signed.json").read_text(encoding="utf-8"))
    n = len(ab["rows"])
    s = ab["summary"]
    purity = [s["signed"]["avg_purity"] / n, s["unsigned"]["avg_purity"] / n,
              s["similarity"]["avg_purity"] / n]
    conflict = [s["signed"]["conflict_ratio"] / n,
                s["unsigned"]["conflict_ratio"] / n,
                s["similarity"]["conflict_ratio"] / n]
    p3 = json.loads((DATA / "eval_p3.json").read_text(encoding="utf-8"))
    auc = [p3["risk_auc"]["learned"], p3["risk_auc"]["control"]]
    labels = ["Signed graph", "Unsigned", "Similarity"]
    colors = [BLUE, PURP, GRAY]
    hatches = [None, "///", "xx"]

    fig, axes = plt.subplots(1, 3, figsize=(152 / 25.4, 46 / 25.4),
                             layout="constrained")
    panels = [
        (axes[0], purity, "Coalition purity", "n = 90 queries"),
        (axes[1], conflict, "Conflict-in-coalition", None),
        (axes[2], auc, "Held-out AUC", "n = 70 queries"),
    ]
    for ax, vals, title, note in panels:
        xlab = labels if len(vals) == 3 else ["Learned MLP", "Rule baseline"]
        cs = colors if len(vals) == 3 else [BLUE, VERM]
        hs = hatches if len(vals) == 3 else [None, "///"]
        b = ax.bar(xlab, vals, color=cs, width=0.58, edgecolor="black",
                   linewidth=0.6)
        for rect, h, v in zip(b, hs, vals):
            if h:
                rect.set_hatch(h)
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.03,
                    f"{v:.3f}", ha="center", fontsize=6.2)
        ax.set_title(title, fontsize=6.8)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=6.2)
        ax.spines[["top", "right"]].set_visible(False)
        if note:
            ax.text(0.02, 0.965, note, transform=ax.transAxes, fontsize=5.6,
                    color=GRAYT, va="top")
    save(fig, "fig3_ablation")


# ---------------------------------------------------------------- Fig. 4 τ 扫描
def fig4():
    p3 = json.loads((DATA / "eval_p3.json").read_text(encoding="utf-8"))
    taus = p3["config"]["taus"]
    rows = [p3["defended"][f"tau={t}"] for t in taus]
    asr = [r["asr"] for r in rows]
    acc = [r["correct"] for r in rows]
    ref = [r["refuse"] for r in rows]
    base_asr = p3["baseline"]["asr"]

    fig, ax = plt.subplots(figsize=(152 / 25.4, 46 / 25.4),
                           layout="constrained")
    ax.plot(taus, asr, marker="s", ms=4, color=VERM, lw=1.1,
            label="Defended ASR")
    ax.plot(taus, acc, marker="o", ms=4, color=BLUE, lw=1.1,
            label="Answer accuracy")
    ax.plot(taus, ref, marker="x", ms=4, color=PURP, lw=1.1, ls=":",
            label="Refusal rate")
    ax.axhline(base_asr, color=BLACK, ls="--", lw=0.9,
               label=f"Undefended ASR ({base_asr:.3f})")
    ax.set_xlabel("Risk threshold  $\\tau$", fontsize=7)
    ax.set_ylabel("Rate", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.2, frameon=False, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.97, "n = 70 held-out queries", transform=ax.transAxes,
            fontsize=5.8, color=GRAYT, va="top", ha="right")
    save(fig, "fig4_tau_sensitivity")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    manifest = {
        "produced": "scripts/make_figs.py",
        "raw_data": {
            "fig3": ["output/eval_pipeline/ablation_signed.json "
                     "(summary keys are SUMS over 90 rows; divided by n=90 "
                     "matches reported per-query means)",
                     "output/eval_pipeline/eval_p3.json risk_auc"],
            "fig4": ["output/eval_pipeline/eval_p3.json "
                     "defended[tau=*]{asr,correct,refuse}, baseline.asr"],
        },
        "transformations": ["ablation summary divided by n=90",
                            "fig4 plots measured values, no smoothing"],
        "missing": "n/a",
    }
    (FIG_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("manifest -> figures/manifest.json")
