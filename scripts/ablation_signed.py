"""P1 消融预实验：有符号图 vs 无符号图 vs 纯相似度图（联盟发现效果对比）。

核心假设：投毒 claims 互相支持（正边）且与真实 claims 矛盾（负边），因此：
    - 有符号图：正边连通分量 → 纯净的投毒联盟/正常联盟（纯度最高），
      负边暴露"反驳联盟"（conflict 信号）
    - 无符号图：正负边混入同一连通块 → 联盟掺入对立 claims（纯度下降）
    - 纯相似度图：反驳对文本高度相似 → 错误合并（纯度最差）

指标（每 query，每模式）：
    n_coalitions           联盟数（投毒 claims 应聚成少数几个联盟）
    avg_purity             联盟平均纯度（max(投毒比, 正常比)）
    poisoned_coalition_ratio  投毒占比 >=0.8 的联盟比例
    conflict_ratio         含内部负边（矛盾）的联盟比例

用法：
    python scripts/ablation_signed.py --dataset nq --num_queries 30 [--no-llm]
    LLM_BACKEND=vllm python scripts/ablation_signed.py --dataset all --num_queries 100
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import INDEX_DIR, OUTPUT_DIR, TOP_K
from graph import ClaimGraph, similarity_graph
from llm import get_backend
from relation import LLMJudge, NLIScorer, RelationExtractor
from retrieval import Embedder, KnowledgeBase
from retrieval.kb import load_poisoned_docs


def synthetic_normal_claims(item: dict, n: int) -> list[dict]:
    """从正确答案构造 n 条'正常' claim（迷你 KB 无正常语料时制造矛盾对）。"""
    q = item["question"].rstrip("?").lower()
    ans = item["correct answer"]
    template = (
        "According to the official records, the answer to \"{q}\" is {ans}, "
        "and this is confirmed by multiple independent sources."
    )
    return [
        {"claim_id": f"normal-{i}", "text": template.format(q=q[:60], ans=ans)[:220],
         "meta": {"poisoned": False}}
        for i in range(n)
    ]


def run_one(kb, embedder, q: str, item: dict, extractor, synth_normal: int) -> dict:
    docs = kb.search(q, top_k=TOP_K)
    claims = []
    for d in docs:
        poisoned_flag = ":poisoned:" in d["doc_id"]
        parts = d["doc_id"].split(":")
        cid = f"{parts[-2][:8]}-{len(claims)}"
        claims.append({"claim_id": cid, "text": d["text"][:220],
                       "meta": {"poisoned": poisoned_flag}})
    claims += synthetic_normal_claims(item, synth_normal)

    edges = extractor.extract_all(claims)
    g = ClaimGraph(claims, edges)
    g_sim = similarity_graph(claims, embedder, threshold=0.6)

    return {
        "signed": g.summary("signed"),
        "unsigned": g.summary("unsigned"),
        "similarity": g_sim.summary("unsigned"),
    }


def main(datasets: list[str], num_queries: int, use_llm: bool, synth_normal: int):
    embedder = Embedder()
    nli = NLIScorer()
    judge = LLMJudge(get_backend()) if use_llm else None
    extractor = RelationExtractor(nli, judge)

    agg = {m: {"n_coalitions": 0.0, "avg_purity": 0.0,
               "poisoned_coalition_ratio": 0.0, "conflict_ratio": 0.0}
           for m in ("signed", "unsigned", "similarity")}
    n_total = 0
    rows = []

    for ds in datasets:
        kb_root = INDEX_DIR / f"{ds}_kb"
        kb = (KnowledgeBase.load(f"{ds}_kb", embedder) if kb_root.exists()
              else __import__("retrieval.kb", fromlist=["build_mini_kb"]).build_mini_kb(embedder, ds))
        poisoned = load_poisoned_docs(ds)
        for qid, item in list(poisoned.items())[:num_queries]:
            r = run_one(kb, embedder, item["question"], item, extractor, synth_normal)
            rows.append({"dataset": ds, "query_id": qid, **r})
            n_total += 1
            for m in agg:
                for k in agg[m]:
                    agg[m][k] += r[m][k]

    print("=" * 72)
    print(f"消融预实验 | datasets={datasets} | queries/数据集={num_queries} | "
          f"总计 {n_total} | LLM兜底={use_llm} | 合成正常claims={synth_normal}")
    print("=" * 72)
    print(f"{'模式':<12}{'联盟数':>8}{'平均纯度':>10}{'投毒联盟占比':>14}{'冲突联盟占比':>14}")
    for m in ("signed", "unsigned", "similarity"):
        n = max(n_total, 1)
        print(f"{m:<12}{agg[m]['n_coalitions']/n:>8.2f}"
              f"{agg[m]['avg_purity']/n:>10.3f}"
              f"{agg[m]['poisoned_coalition_ratio']/n:>14.3f}"
              f"{agg[m]['conflict_ratio']/n:>14.3f}")

    out_path = OUTPUT_DIR / "eval_pipeline" / "ablation_signed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": agg, "rows": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ablation] saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", help="all | nq,hotpotqa,msmarco")
    ap.add_argument("--num_queries", type=int, default=30)
    ap.add_argument("--no-llm", action="store_true", help="禁用 LLM 兜底（纯 NLI）")
    ap.add_argument("--synthetic-normal", type=int, default=2,
                    help="混入的合成正常 claims 数（全量 KB 时设 0）")
    args = ap.parse_args()
    ds = ["nq", "hotpotqa", "msmarco"] if args.dataset == "all" \
        else [d.strip() for d in args.dataset.split(",")]
    main(ds, args.num_queries, use_llm=not args.no_llm,
         synth_normal=args.synthetic_normal)
