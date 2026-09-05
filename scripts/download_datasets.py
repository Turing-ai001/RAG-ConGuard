"""下载 BEIR 数据集子集（官方 zip 源），存到 data/raw/。

HF 的 datasets 5.x 已移除 script-based loader，BeIR/beir 仓库只剩 beir.py
无法通过 Hub 加载（streaming 亦异常）；故直接从 BEIR 官方发布源（UKP 公开链接）
下载 zip，行结构与 Hub 的 jsonl 完全一致，保证与旧项目语料可比。

用法：
    python scripts/download_datasets.py [--datasets nq,hotpotqa,msmarco] [--corpus 600000] [--queries 5000]

说明：
    - 下载 {BASE}/{name}.zip 到临时文件，只取前 --corpus 条语料
    - queries 取前 --queries 条
    - 产物：data/raw/{name}_corpus.jsonl + data/raw/{name}_queries.jsonl（与旧脚本一致）
"""
import argparse
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import RAW_DIR

BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def _download(url: str, dest: Path) -> None:
    """带重试的下载（失败可重跑）。"""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as resp, \
                 open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
            print(f"[download] {dest.name} {dest.stat().st_size / 1e6:.0f} MB")
            return
        except Exception as e:
            print(f"[download] attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    raise RuntimeError(f"download failed: {url}")


def download(dataset: str, corpus_n: int, query_n: int) -> None:
    corpus_path = RAW_DIR / f"{dataset}_corpus.jsonl"
    queries_path = RAW_DIR / f"{dataset}_queries.jsonl"
    print(f"[download] {dataset}: corpus<= {corpus_n}, queries<= {query_n}")

    zip_path = Path(tempfile.gettempdir()) / f"beir_{dataset}.zip"
    _download(f"{BASE}/{dataset}.zip", zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"{dataset}/corpus.jsonl") as src, \
             open(corpus_path, "wb") as out:
            n = 0
            for line in src:
                if n >= corpus_n:
                    break
                out.write(line)
                n += 1
        with zf.open(f"{dataset}/queries.jsonl") as src, \
             open(queries_path, "wb") as out:
            m = 0
            for line in src:
                if m >= query_n:
                    break
                out.write(line)
                m += 1
    zip_path.unlink()
    print(f"[download] {dataset} done -> corpus {n} 行, queries {m} 行")


if __name__ == "__main__":
    from config import CORPUS_SUBSET, DEFAULT_DATASETS, QUERY_SUBSET
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--corpus", type=int, default=CORPUS_SUBSET)
    ap.add_argument("--queries", type=int, default=QUERY_SUBSET)
    args = ap.parse_args()
    for name in args.datasets.split(","):
        download(name.strip(), args.corpus, args.queries)
