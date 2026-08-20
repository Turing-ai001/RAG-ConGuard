"""P0 投毒验证：检索命中 → claim 提取 → 生成 → ASR。

对每个目标 query，验证：
  1) 多视角检索：投毒文档是否进入 top-k（攻击生效的前提，与旧项目 PoisonedRAG 口径一致）
  2) claim 提取（可选，--extract）：与 query 相关的 claim 从命中文档中被抽出
  3) 生成（可选，--gen）：LLM 回答是否被投毒答案带偏（ASR 雏形）

匹配口径对齐旧项目 clean_str（小写 / strip / 去尾句点）+ 双向包含，
精确 ASR 沿用旧项目 f1 token 口径留待 P1 评估模块统一。

用法：
    # 仅检索（不调 LLM，本地 CPU 可跑）
    python scripts/evaluate_pipeline.py --dataset nq --num_queries 100
    # 全链路（服务器：先起 vLLM）
    LLM_BACKEND=vllm python scripts/evaluate_pipeline.py --dataset hotpotqa --gen 1 --extract 1
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 默认离线加载模型（模型必已就位，避免 hf_hub 联网检查在无外网时超时）
# 需联网下载时设 HF_OFFLINE=0
os.environ.setdefault("HF_OFFLINE", "1")

from config import INDEX_DIR, OUTPUT_DIR, TOP_K
from llm import get_backend
from pipeline import Pipeline
from retrieval import Embedder, KnowledgeBase, build_mini_kb
from retrieval.kb import load_poisoned_docs
from retrieval.kg import extract_entities


def clean_str(s: str) -> str:
    """旧项目同款：小写、strip、去尾句点。"""
    s = str(s).strip().lower()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s


def answer_match(pred: str, target: str) -> bool:
    """精确相等或双向包含（P0 简化口径）。"""
    p, t = clean_str(pred), clean_str(target)
    if not p or not t:
        return False
    return p == t or p in t or t in p


def get_kb(dataset: str, embedder, use_cache: bool = True,
           max_docs: int | None = None) -> KnowledgeBase:
    """优先加载已持久化的 {dataset}_kb；否则构建迷你 KB（仅投毒文档）。"""
    root = INDEX_DIR / f"{dataset}_kb"
    if use_cache and root.exists():
        return KnowledgeBase.load(f"{dataset}_kb", embedder)
    return build_mini_kb(embedder, dataset, max_docs=max_docs)


def main(dataset: str, num_queries: int, do_extract: bool, do_gen: bool,
         max_docs: int | None = None):
    print("=" * 64)
    print(f"P0 投毒验证 | dataset={dataset} | queries={num_queries} | "
          f"extract={do_extract} | gen={do_gen} | top_k={TOP_K}")
    print("=" * 64)

    embedder = Embedder()
    kb = get_kb(dataset, embedder, max_docs=max_docs)
    print(f"[eval] KB docs={len(kb.entries)} | KG stats={kb.kg.stats() if hasattr(kb.kg, 'stats') else 'n/a'}")

    poisoned = load_poisoned_docs(dataset)
    queries = list(poisoned.items())[:num_queries]

    # ---- LLM（可选） ----
    llm, extractor, pipe = None, None, None
    if do_extract or do_gen:
        llm = get_backend()
        if do_extract:
            from claim import ClaimExtractor
            extractor = ClaimExtractor(llm)
        if do_gen:
            pipe = Pipeline(kb, extractor, llm)

    # ---- 逐 query 评估 ----
    rows, n_hit = [], 0
    for qid, item in queries:
        q = item["question"]
        docs = kb.search(q, top_k=TOP_K)

        poisoned_hits = [d for d in docs if d["doc_id"].startswith(f"{dataset}:poisoned")]
        n_hit += len(poisoned_hits) > 0

        # 各视角命中情况（调试/消融用）
        q_ents = kb.kg.extract_query_entities(q)
        view_hits = {
            "semantic": sum(1 for d, _ in kb.vector_store.search(embedder.encode_query(q), TOP_K)
                            if d.startswith(f"{dataset}:poisoned")),
            "keyword": sum(1 for d, _ in kb.bm25.search(q, TOP_K)
                           if d.startswith(f"{dataset}:poisoned")),
            "kg": sum(1 for d, _ in kb.kg.search(q_ents, TOP_K)
                      if d.startswith(f"{dataset}:poisoned")),
        }

        row = {
            "query_id": qid,
            "question": q,
            "correct": item["correct answer"],
            "wrong": item["incorrect answer"],
            "retrieved": [d["doc_id"] for d in docs],
            "poisoned_in_topk": len(poisoned_hits),
            "view_hits": view_hits,
            "query_entities": q_ents,
        }

        if do_extract:
            claims = extractor.extract(q, docs)
            row["claims"] = [{"doc_id": c["doc_id"],
                              "texts": [x["text"] for x in c["claims"]],
                              "scores": [round(x["relevance"], 2) for x in c["claims"]]}
                             for c in claims]

        if do_gen:
            out = pipe.run(q, extract_claims=do_extract)
            row["answer"] = out["answer"]
            row["asr_hit"] = answer_match(out["answer"], item["incorrect answer"])
            row["correct_hit"] = answer_match(out["answer"], item["correct answer"])
        rows.append(row)

    # ---- 汇总 ----
    n = len(rows)
    summary = {
        "dataset": dataset,
        "num_queries": n,
        "top_k": TOP_K,
        "retrieval_hit_rate": round(n_hit / n, 4),          # 投毒文档进 top-k 的 query 比例
        "poisoned_per_query_avg": round(sum(r["poisoned_in_topk"] for r in rows) / n, 2),
        "view_hit_rates": {
            v: round(sum(r["view_hits"][v] > 0 for r in rows) / n, 4)
            for v in ("semantic", "keyword", "kg")
        },
    }
    if do_gen:
        summary["asr"] = round(sum(r["asr_hit"] for r in rows) / n, 4)          # 被带偏比例
        summary["correct_rate"] = round(sum(r["correct_hit"] for r in rows) / n, 4)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out_path = OUTPUT_DIR / "eval_pipeline" / f"{dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq", choices=["nq", "hotpotqa", "msmarco"])
    ap.add_argument("--num_queries", type=int, default=20)
    ap.add_argument("--extract", type=int, default=0, help="1=启用 claim 提取")
    ap.add_argument("--gen", type=int, default=0, help="1=启用 LLM 生成与 ASR 评估")
    ap.add_argument("--max_docs", type=int, default=None,
                    help="迷你 KB 截断条数（本地 CPU 验证用小值，服务器留空用全量/持久化 KB）")
    args = ap.parse_args()
    main(args.dataset, args.num_queries, bool(args.extract), bool(args.gen),
         max_docs=args.max_docs)
