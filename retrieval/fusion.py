"""多视角检索结果融合：RRF（Reciprocal Rank Fusion）。

每个视角返回按分降序的 [(doc_id, score)]，融合分数：
    score(d) = Σ_view weight_view / (RRF_K + rank_view(d) + 1)
rank 从 0 计。RRF 只依赖排序不依赖分数量纲，适配 BM25/余弦/图分 三类异构分数。
"""
from config import FUSION_POOL_K, FUSION_RRF_K, FUSION_WEIGHTS, TOP_K

ViewResults = dict[str, list[tuple[str, float]]]  # view_name -> [(doc_id, score)]


def rrf_fuse(
    view_results: ViewResults,
    top_k: int = TOP_K,
    pool_k: int = FUSION_POOL_K,
    k: int = FUSION_RRF_K,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """融合多个视角 → [(doc_id, rrf_score), ...] 降序，取前 top_k。"""
    w = {**FUSION_WEIGHTS, **(weights or {})}
    fused: dict[str, float] = {}
    for view, items in view_results.items():
        wgt = w.get(view, 1.0)
        for rank, (doc_id, _score) in enumerate(items[:pool_k]):
            fused[doc_id] = fused.get(doc_id, 0.0) + wgt / (k + rank + 1)
    ranked = sorted(fused.items(), key=lambda x: -x[1])
    return ranked[:top_k]
