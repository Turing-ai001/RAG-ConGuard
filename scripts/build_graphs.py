"""P1 图构建 + 可视化（人工检查 20-30 case）。

链路：query → 多视角检索（P0）→ claim 提取（P0）→ 关系抽取（P1，NLI 主干+LLM 兜底）
      → 有符号图 → 可视化 PNG + 边列表 txt

用法：
    # 本机小规模（NLI 模型 + LLM 均已就位）
    python scripts/build_graphs.py --dataset nq --num_queries 5
    # 服务器全量（7B judge）
    LLM_BACKEND=vllm python scripts/build_graphs.py --dataset nq --num_queries 25
    # 纯 NLI（无 LLM 兜底，消融对比用）
    python scripts/build_graphs.py --dataset hotpotqa --num_queries 10 --no-llm

产物：output/graphs/{dataset}_{qid}.png + _edges.txt
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import INDEX_DIR, TOP_K
from claim import ClaimExtractor
from graph import ClaimGraph, draw_graph
from llm import get_backend
from relation import LLMJudge, NLIScorer, RelationExtractor
from retrieval import Embedder, KnowledgeBase
from retrieval.kb import load_poisoned_docs
from retrieval.kg import extract_entities


def doc_id_to_claim(doc_id: str) -> str:
    """文档 doc_id → claim 前缀（doc_id 较长，图节点用 qid 段）。"""
    parts = doc_id.split(":")
    return parts[-2][:8] if len(parts) >= 3 else doc_id[:8]


def synthetic_normal_claims(item: dict, n: int) -> list[dict]:
    """从投毒数据的正确答案构造 n 条'正常' claim（模拟真实 KB 中的正常文档）。

    迷你 KB 只有投毒文档时，图里没有矛盾对，无法验证反驳方向；
    混入正确答案 claims 后：投毒(+错误答案) ↔ 正常(+正确答案) 应判 contradiction。
    服务器全量 KB（含真实语料）时用 --synthetic-normal 0。
    """
    q = item["question"].rstrip("?").lower()
    ans = item["correct answer"]
    template = (
        "According to the official records, the answer to \"{q}\" is {ans}, "
        "and this is confirmed by multiple independent sources."
    )
    return [
        {"claim_id": f"normal-{i}", "text": template.format(q=q[:60], ans=ans)[:220],
         "meta": {"poisoned": False, "doc_id": "synthetic:normal"}}
        for i in range(n)
    ]


def main(dataset: str, num_queries: int, use_llm: bool, synth_normal: int):
    print("=" * 70)
    print(f"P1 图构建 | dataset={dataset} | queries={num_queries} | "
          f"LLM兜底={use_llm} | 合成正常claims={synth_normal} | top_k={TOP_K}")
    print("=" * 70)

    embedder = Embedder()
    # 优先已持久化 KB（服务器构建过 {dataset}_kb）；否则迷你 KB
    kb_root = INDEX_DIR / f"{dataset}_kb"
    kb = (KnowledgeBase.load(f"{dataset}_kb", embedder) if kb_root.exists()
          else __import__("retrieval.kb", fromlist=["build_mini_kb"]).build_mini_kb(embedder, dataset))
    print(f"[graphs] KB docs={len(kb.entries)}")

    nli = NLIScorer()
    llm = None
    judge = None
    if use_llm:
        llm = get_backend()
        judge = LLMJudge(llm)
    extractor = RelationExtractor(nli, judge)

    poisoned = load_poisoned_docs(dataset)
    queries = list(poisoned.items())[:num_queries]

    for qid, item in queries:
        q = item["question"]
        docs = kb.search(q, top_k=TOP_K)

        # 1) claims（含来源 meta；混入合成正常 claims 以产生矛盾对）
        claims: list[dict] = []
        for d in docs:
            poisoned_flag = d["doc_id"].startswith(f"{dataset}:poisoned")
            cid = f"{doc_id_to_claim(d['doc_id'])}-{len(claims)}"
            claims.append({"claim_id": cid, "text": d["text"][:220],
                           "meta": {"poisoned": poisoned_flag,
                                    "doc_id": d["doc_id"]}})
        claims += synthetic_normal_claims(item, synth_normal)

        # 2) 关系抽取（NLI 主干；LLM 兜底可选）
        edges = extractor.extract_all(claims)
        stats = extractor.stats(edges)

        # 3) 有符号图 + 联盟
        g = ClaimGraph(claims, edges)
        summ = g.summary("signed")

        # 4) 可视化
        title = f"[{qid}] {q[:70]}\n{len(claims)} claims / {len(edges)} edges / " \
                f"LLM兜底 {stats['llm_fallback_ratio']:.0%} / " \
                f"联盟{summ['n_coalitions']}个 纯度{summ['avg_purity']}"
        draw_graph(g, f"{dataset}_{qid}", title=title)

        print(f"  [{qid}] {q[:60]}...")
        print(f"       claims={len(claims)} edges={len(edges)} "
              f"(nli={len(edges) - stats['llm_fallback_ratio'] * len(edges):.0f}"
              f"/llm={stats['llm_fallback_ratio'] * len(edges):.0f}) "
              f"by_type={stats['by_type']}")
        print(f"       联盟: {summ['n_coalitions']}个 | 平均纯度 {summ['avg_purity']} | "
              f"投毒联盟占比 {summ['poisoned_coalition_ratio']}")

    print(f"\n[graphs] 完成：检查 output/graphs/{dataset}_*.png + *_edges.txt")
    print("人工核对要点：支持/反驳方向、投毒 claim 联盟是否独立成簇")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq", choices=["nq", "hotpotqa", "msmarco"])
    ap.add_argument("--num_queries", type=int, default=5)
    ap.add_argument("--no-llm", action="store_true", help="禁用 LLM 兜底（纯 NLI）")
    ap.add_argument("--synthetic-normal", type=int, default=2,
                    help="混入的合成正常 claims 数（产生矛盾对；全量 KB 时设 0）")
    args = ap.parse_args()
    main(args.dataset, args.num_queries, use_llm=not args.no_llm,
         synth_normal=args.synthetic_normal)
