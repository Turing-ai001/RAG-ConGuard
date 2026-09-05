"""构建知识库（原始语料子集 + 投毒文档）→ 向量索引 + BM25，持久化到 data/index/{name}。

用法：
    python scripts/build_kb.py [--datasets nq,hotpotqa]

依赖：先跑 scripts/download_datasets.py 生成 data/raw/{dataset}_corpus.jsonl。
投毒文档从 data/poisoned/{dataset}.json 读入（已复用旧项目数据）。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import RAW_DIR, TOP_K
from retrieval.embedder import Embedder
from retrieval.kb import KnowledgeBase, poisoned_docs_to_entries
from retrieval.kg import EmptyKG, default_kg_store


def build(dataset: str, embedder: Embedder, use_kg: bool = True) -> KnowledgeBase:
    corpus_path = RAW_DIR / f"{dataset}_corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} 不存在，请先运行 scripts/download_datasets.py"
        )

    entries = []
    with open(corpus_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            entries.append({
                "doc_id": f"{dataset}:corpus:{i}",
                "text": row.get("text", ""),
                "meta": {"source": "corpus", "dataset": dataset},
            })
    entries += poisoned_docs_to_entries(dataset)

    # --no-kg：跳过 KG 视角构建（实体抽取正则存在嵌套量词灾难回溯，
    #        600k 文本上过慢；先用 semantic+keyword 两视角跑主线实验，
    #        KG 修复后另行重建补全）
    kg_store = default_kg_store() if not use_kg else None
    kb = KnowledgeBase(f"{dataset}_kb", entries, embedder,
                       kg_store=kg_store or EmptyKG())
    kb.save()
    print(f"[build_kb] {dataset}: {len(entries)} docs ({len(entries) - len(poisoned_docs_to_entries(dataset))} corpus + poisoned)")
    return kb


if __name__ == "__main__":
    from config import DEFAULT_DATASETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--no-kg", action="store_true",
                    help="跳过 KG 视角构建（应急：实体正则回溯过慢）")
    args = ap.parse_args()
    embedder = Embedder()
    for name in args.datasets.split(","):
        build(name.strip(), embedder, use_kg=not args.no_kg)
