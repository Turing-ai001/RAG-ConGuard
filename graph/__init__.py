"""Claim 图模块（P1）：节点=claim、边=关系（有符号图 + 三种变体 + 联盟发现）。"""
from graph.build import ClaimGraph, signed_semantic_graph, similarity_graph


def draw_graph(*args, **kwargs):
    """惰性导入：visualize 依赖 matplotlib/networkx，仅真正画图时加载，
    服务器等无 matplotlib 环境仍可跑图构建/评估。"""
    from graph.visualize import draw_graph as _draw
    return _draw(*args, **kwargs)


__all__ = ["ClaimGraph", "similarity_graph", "draw_graph"]
