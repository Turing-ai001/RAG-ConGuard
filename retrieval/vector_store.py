"""FAISS 向量库：IndexFlatIP（内积）+ 归一化 → 余弦相似度检索。
支持持久化到磁盘（.faiss 索引 + ids.json），可增量 add。
"""
import json
import os
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []
        self.id2pos: dict[str, int] = {}

    # ---------- 构建 ----------
    def add(self, doc_ids: list[str], embeddings: np.ndarray) -> None:
        """添加一批 (doc_id, embedding)。embedding 会被就地 L2 归一化。"""
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            raise ValueError(f"embedding shape mismatch: {emb.shape}, expected (n, {self.dim})")
        faiss.normalize_L2(emb)
        start = self.index.ntotal
        self.index.add(emb)
        for i, doc_id in enumerate(doc_ids):
            if doc_id in self.id2pos:
                raise ValueError(f"duplicate doc_id: {doc_id}")
            self.id2pos[doc_id] = start + i
            self.ids.append(doc_id)

    def __len__(self) -> int:
        return self.index.ntotal

    # ---------- 检索 ----------
    def search(self, query_emb: np.ndarray, k: int) -> list[tuple[str, float]]:
        """返回 [(doc_id, cosine_score), ...] 降序。"""
        q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k)
        return [
            (self.ids[i], float(s))
            for i, s in zip(idxs[0], scores[0])
            if 0 <= i < len(self.ids)
        ]

    # ---------- 持久化 ----------
    # 注意：faiss C 层用 fopen（Windows 上按 ANSI 代码页解析路径），
    # 中文/非 ASCII 目录（如 F:\论文\RAG）会打不开 —— 统一 chdir 后使用相对文件名。
    def save(self, index_path: Path | str, ids_path: Path | str) -> None:
        index_path = Path(index_path)
        ids_path = Path(ids_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(index_path.parent)
        try:
            faiss.write_index(self.index, index_path.name)
        finally:
            os.chdir(cwd)
        ids_path.write_text(json.dumps(self.ids, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path | str, ids_path: Path | str) -> "VectorStore":
        index_path = Path(index_path)
        ids_path = Path(ids_path)
        vs = cls.__new__(cls)
        cwd = os.getcwd()
        os.chdir(index_path.parent)
        try:
            vs.index = faiss.read_index(index_path.name)
        finally:
            os.chdir(cwd)
        vs.dim = vs.index.d
        vs.ids = json.loads(ids_path.read_text(encoding="utf-8"))
        vs.id2pos = {doc_id: i for i, doc_id in enumerate(vs.ids)}
        return vs
