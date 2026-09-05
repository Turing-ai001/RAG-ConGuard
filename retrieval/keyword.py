"""BM25 关键词检索（rank_bm25）。
视角 2：仅返回与 query 有词项重叠的文档（0 分文档不参与融合，避免 RRF 噪声）。
"""
import multiprocessing as mp
import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_N_WORKERS = 16   # 全量 60 万文档 tokenize 单线程受限（正则扫描），并行化构建


def tokenize(text: str) -> list[str]:
    """简单词元化：小写 + 字母数字。P0 够用，后续可换更专业的分词。"""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, doc_ids: list[str], texts: list[str]):
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have equal length")
        self.doc_ids = list(doc_ids)
        if len(texts) < 10000:                     # 小规模：直接单进程（进程池开销大于收益）
            corpus = [tokenize(t) for t in texts]
        else:
            with mp.Pool(processes=_N_WORKERS) as pool:
                corpus = list(pool.imap(tokenize, texts, chunksize=512))
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """返回 [(doc_id, bm25_score), ...]，仅含命中词项重叠的文档。"""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)
        hits = np.where(scores > 0.0)[0]
        if hits.size == 0:
            return []
        top = hits[np.argsort(scores[hits])[-k:][::-1]]
        return [(self.doc_ids[i], float(scores[i])) for i in top]

    def search_verbose(self, query: str, k: int) -> list[tuple[str, float, list[str]]]:
        """带命中的词项（调试用）。"""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        hits = np.where(scores > 0.0)[0]
        top = hits[np.argsort(scores[hits])[-k:][::-1]]
        out = []
        for i in top:
            matched = [t for t in tokens if t in self.bm25.corpus[i]]
            out.append((self.doc_ids[i], float(scores[i]), matched))
        return out
