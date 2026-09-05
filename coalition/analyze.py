"""联盟特征（P2 前半）：区分"证据联盟"与"投毒联盟"的结构信号。

正常证据联盟：内聚（互相支持）+ 外部佐证（被其他来源支持）；
投毒联盟：     高度内聚（互证） + 外部佐证缺失（孤立） + 常被外部反驳。

指标（均按联盟大小归一，跨 query 可比）：
    cohesion               内部正边密度 = internal_pos / C(size, 2)
    external_corroboration 每个成员平均外部正边 = external_pos / size（0=完全孤立）
    external_refute        每个成员平均外部负边 = external_neg / size
"""
from __future__ import annotations

from math import comb

from graph.build import ClaimGraph


def coalition_features(g: ClaimGraph, coalitions: list[set[str]]) -> list[dict]:
    """每个联盟的结构特征。coalitions 由 g.coalition_nodes() 给出。

    边按无向处理（两端一内一外跨联盟），与联盟发现语义一致。
    """
    pos_set = _undirected(g.pos_edges)
    neg_set = _undirected(g.neg_edges)
    out = []
    for comp in coalitions:
        size = len(comp)
        if size == 0:
            continue
        internal_pos = _count_inside(pos_set, comp)
        internal_neg = _count_inside(neg_set, comp)
        external_pos = _count_cross(pos_set, comp)
        external_neg = _count_cross(neg_set, comp)
        denom = comb(size, 2) if size > 1 else 1.0
        out.append({
            "size": size,
            "cohesion": round(internal_pos / denom, 4),
            "internal_pos": internal_pos,
            "internal_neg": internal_neg,
            "external_corroboration": round(external_pos / size, 4),
            "external_refute": round(external_neg / size, 4),
            "external_pos": external_pos,
            "external_neg": external_neg,
            "claims": sorted(comp),
        })
    return out


def _undirected(edges: list[tuple[str, str]]) -> set[tuple[str, str]]:
    """有向边集合 → 无向化去重：{(a,b)} 统一 (min, max) 排序。"""
    return {(min(a, b), max(a, b)) for a, b in edges}


def _count_inside(undirected: set[tuple[str, str]], comp: set[str]) -> int:
    return sum(1 for a, b in undirected if a in comp and b in comp)


def _count_cross(undirected: set[tuple[str, str]], comp: set[str]) -> int:
    return sum(1 for a, b in undirected if (a in comp) + (b in comp) == 1)
