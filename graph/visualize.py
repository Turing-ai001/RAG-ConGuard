"""图可视化：networkx + matplotlib 导出 PNG（人工检查用）。

约定：
    节点颜色  橙=投毒 claim，蓝=正常 claim（meta.poisoned）
    边颜色    绿=支持(+)，红=反驳(−)，灰=无关联，虚线=LLM 兜底来源
    边标签    关系类型（省略 UNRELATED 的标签，只画灰线）
输出：output/graphs/{name}.png + 同名的边列表 txt（文本核对方向用）。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示环境（服务器）也可出图
import matplotlib.pyplot as plt
import networkx as nx

from config import OUTPUT_DIR
from relation.types import RelationType

# 中文字体（Windows: 微软雅黑/黑体；Linux 服务器: Noto CJK），缺失时回退默认（仅 title 变方块，不崩）
from matplotlib import font_manager

_FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                    "WenQuanYi Zen Hei", "Arial Unicode MS"]
for _f in font_manager.fontManager.ttflist:
    if any(c in _f.name for c in _FONT_CANDIDATES):
        plt.rcParams["font.sans-serif"] = [_f.name]
        break
plt.rcParams["axes.unicode_minus"] = False

EDGE_COLOR = {
    RelationType.SUPPORT: "#2e8b57",
    RelationType.ENTAIL: "#2e8b57",
    RelationType.REFUTE: "#c0392b",
    RelationType.CONTRADICT: "#c0392b",
    RelationType.UNRELATED: "#b0b0b0",
}


def draw_graph(g, name: str, out_dir: Path | None = None,
               title: str = "", verbose: bool = True) -> Path:
    """渲染 ClaimGraph → PNG + 边列表 txt。返回 PNG 路径。"""
    out_dir = out_dir or OUTPUT_DIR / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    node_colors = []
    for c in g.claims:
        node_colors.append("#e67e22" if c.get("meta", {}).get("poisoned")
                           else "#3498db")

    edges, ecolors, elabels = [], [], {}
    for r in g.relations:
        edges.append((r.src, r.dst))
        ecolors.append(EDGE_COLOR[r.type])
        if r.type is not RelationType.UNRELATED:
            elabels[(r.src, r.dst)] = r.type.label

    G = nx.DiGraph()
    G.add_nodes_from(g.claim_ids)
    G.add_edges_from(edges)
    try:
        pos = nx.spring_layout(G, seed=42, k=0.6)
    except Exception:
        pos = nx.circular_layout(G)

    plt.figure(figsize=(12, 9))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900, alpha=0.95)
    nx.draw_networkx_edges(G, pos, edge_color=ecolors, arrows=True,
                           arrowstyle="-|>", arrowsize=14, width=1.8,
                           connectionstyle="arc3,rad=0.12")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=elabels, font_size=7)
    nx.draw_networkx_labels(G, pos, font_size=6.5)

    # 节点下方标注来源（投毒/正常）
    labels = {}
    for c in g.claims:
        tag = "P" if c.get("meta", {}).get("poisoned") else "N"
        labels[c["claim_id"]] = f"{c['claim_id']}\n[{tag}]"
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6,
                            verticalalignment="top")

    plt.title(f"{title}\n橙=投毒 蓝=正常 | 绿=支持/蕴含 红=反驳/矛盾 灰=无关联")
    plt.axis("off")
    png_path = out_dir / f"{name}.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=130)
    plt.close()

    # 边列表 txt（文本核对）
    txt_path = out_dir / f"{name}_edges.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        for c in g.claims:
            tag = "POISONED" if c.get("meta", {}).get("poisoned") else "normal"
            f.write(f"# [{c['claim_id']}|{tag}] {c['text']}\n")
        f.write("# --- edges (src -> dst) ---\n")
        for r in sorted(g.relations, key=lambda r: (r.src, r.dst)):
            f.write(f"{r.src} --[{r.type.label}|{r.score:.2f}|{r.provenance}]--> {r.dst}\n")
    if verbose:
        print(f"[graph] saved {png_path}")
    return png_path
