"""下载 BEIR 数据集子集（流式，只取前 N 条语料），存到 data/raw/。

用法：
    python scripts/download_datasets.py [--datasets nq,hotpotqa] [--corpus 2000] [--queries 50]

说明：
    - 流式加载 BeIR/{name} 的 corpus / queries / test 三个配置
    - 语料取前 --corpus 条（本地冒烟用小语料；服务器全量实验把 --corpus 调大）
    - queries 取前 --queries 条
    - 产物：data/raw/{name}_corpus.jsonl + data/raw/{name}_queries.jsonl
"""
import argparse
import json
import itertools

from datasets import load_dataset

from config import RAW_DIR


def download(dataset: str, corpus_n: int, query_n: int) -> None:
    corpus_path = RAW_DIR / f"{dataset}_corpus.jsonl"
    queries_path = RAW_DIR / f"{dataset}_queries.jsonl"

    print(f"[download] {dataset}: corpus<= {corpus_n}, queries<= {query_n}")
    corpus_ds = load_dataset(f"BeIR/{dataset}", "corpus", streaming=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(itertools.islice(corpus_ds, corpus_n)):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[download] corpus -> {corpus_path}")

    # queries/test 两处都取，合并去重（test 即官方测试 query 集）
    seen = set()
    with open(queries_path, "w", encoding="utf-8") as f:
        for split in ("queries", "test"):
            ds = load_dataset(f"BeIR/{dataset}", split, streaming=True)
            for row in itertools.islice(ds, query_n):
                key = row.get("_id") or row.get("query") or str(row)
                if key in seen:
                    continue
                seen.add(key)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[download] queries -> {queries_path}")


if __name__ == "__main__":
    from config import CORPUS_SUBSET, DEFAULT_DATASETS, QUERY_SUBSET
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--corpus", type=int, default=CORPUS_SUBSET)
    ap.add_argument("--queries", type=int, default=QUERY_SUBSET)
    args = ap.parse_args()
    for name in args.datasets.split(","):
        download(name.strip(), args.corpus, args.queries)
