"""Claim 图构建：节点 = claim，边 = claim 间关系（Relation，带类型/极性/置信度）。

三种图变体（消融预实验用）：
    signed      有符号图：边保留极性（+支持 / −反驳），联盟 = 正边连通分量
    unsigned    无符号图：去掉符号（只留"是否相关"），联盟 = 全部相关边连通分量
    similarity  纯相似度图：cosine >= τ 建无向边（无类型无符号）

联盟发现（coalition detection）：
    正边连通分量 = 支持联盟（相互印证的事实组）；
    联盟内负边（矛盾/反驳）数量 = 该联盟的内部冲突信号。

指标（对齐消融口径）：
    purity          联盟纯度：max(投毒占比, 正常占比) ∈ [0.5, 1]
    pos/neg 边数    有符号图区分"支持联盟"与"反驳联盟"的关键
"""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from relation.types import Relation, RelationType


def _connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[set[str]]:
    """无向连通分量（union-find）。"""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)
    groups: dict[str, set[str]] = defaultdict(set)
    for n in nodes:
        groups[find(n)].add(n)
    return list(groups.values())


class ClaimGraph:
    """节点 = claim（含来源 meta），边 = Relation。"""

    def __init__(self, claims: list[dict], relations: list[Relation]):
        self.claims = claims                                  # [{claim_id, text, meta}]
        self.claim_ids = [c["claim_id"] for c in claims]
        self.relations = relations
        self.pos_edges: list[tuple[str, str]] = []            # 极性 + 的边
        self.neg_edges: list[tuple[str, str]] = []            # 极性 - 的边
        self.rel_edges: list[tuple[str, str]] = []            # 所有相关边（非 UNRELATED）
        for r in relations:
            if r.polarity > 0:
                self.pos_edges.append((r.src, r.dst))
            elif r.polarity < 0:
                self.neg_edges.append((r.src, r.dst))
            if r.type is not RelationType.UNRELATED:
                self.rel_edges.append((r.src, r.dst))

    # ---------- 图变体 ----------
    def coalition_nodes(self, mode: str = "signed") -> list[set[str]]:
        """联盟发现（连通分量）：
        signed：只用正边（支持/蕴含）→ 支持联盟；
        unsigned：用全部相关边（正+负）→ 混合连通块；
        similarity：外部构建时传 similarity 边（本方法用 rel_edges 近似）。
        """
        edges = self.pos_edges if mode == "signed" else self.rel_edges
        return _connected_components(self.claim_ids, edges)

    # ---------- 联盟指标 ----------
    def coalition_stats(self, coalitions: list[set[str]]) -> list[dict]:
        """每个联盟：规模 / 内部正负边 / 投毒纯度。"""
        id2poisoned = {c["claim_id"]: bool(c.get("meta", {}).get("poisoned"))
                       for c in self.claims}
        pos_set = {(a, b) for a, b in self.pos_edges}
        neg_set = {(a, b) for a, b in self.neg_edges}
        out = []
        for comp in coalitions:
            if len(comp) < 1:
                continue
            nodes = list(comp)
            pos_n = sum(1 for e in pos_set if e[0] in comp and e[1] in comp)
            neg_n = sum(1 for e in neg_set if e[0] in comp and e[1] in comp)
            poisoned_n = sum(1 for n in nodes if id2poisoned.get(n))
            ratio = poisoned_n / len(nodes)
            out.append({
                "size": len(nodes),
                "pos_edges": pos_n,
                "neg_edges": neg_n,
                "poisoned_ratio": round(ratio, 3),
                "purity": round(max(ratio, 1 - ratio), 3),   # [0.5,1]，1=联盟纯净
                "claims": nodes,
            })
        return out

    def summary(self, mode: str = "signed") -> dict:
        coalitions = self.coalition_nodes(mode)
        stats = self.coalition_stats(coalitions)
        if not stats:
            return {"mode": mode, "n_coalitions": 0, "avg_purity": 0.0,
                    "avg_size": 0.0, "poisoned_coalition_ratio": 0.0}
        n = len(stats)
        return {
            "mode": mode,
            "n_coalitions": n,
            "avg_purity": round(sum(s["purity"] for s in stats) / n, 3),
            "avg_size": round(sum(s["size"] for s in stats) / n, 3),
            "poisoned_coalition_ratio": round(
                sum(1 for s in stats if s["poisoned_ratio"] >= 0.8) / n, 3),
            "conflict_ratio": round(
                sum(1 for s in stats if s["neg_edges"] > 0) / n, 3),
        }


# ---------- 纯相似度图（消融第三种变体） ----------
def similarity_graph(claims: list[dict], embedder, threshold: float = 0.6) -> ClaimGraph:
    """BGE cosine >= τ 建无向边（self 除外），类型标 SUPPORT（仅连通性用）。"""
    texts = [c["text"] for c in claims]
    embs = embedder.encode(texts)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sim = embs @ embs.T
    relations: list[Relation] = []
    ids = [c["claim_id"] for c in claims]
    for i, j in itertools.combinations(range(len(ids)), 2):
        if sim[i, j] >= threshold:
            relations.append(Relation(ids[i], ids[j], RelationType.SUPPORT,
                                      float(sim[i, j]), provenance="similarity"))
    return ClaimGraph(claims, relations)
