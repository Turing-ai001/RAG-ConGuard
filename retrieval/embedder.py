"""Embedding 封装：默认 BAAI/bge-base-en-v1.5（旧项目同款，已验证）。
换 E5 只需改 config.EMBEDDING_MODEL，此处无感知。

离线加载：模型已缓存到本地时传 local_files_only=True（或设 HF_OFFLINE=1），
避免 hf_hub 联网检查超时；服务器首次构建则正常联网下载。
"""
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, DEVICE


class Embedder:
    """统一 embedding 入口：encode(texts) -> L2 归一化的 float32 矩阵。"""

    def __init__(self, model_name: str | None = None,
                 local_files_only: bool | None = None):
        self.model_name = model_name or EMBEDDING_MODEL
        offline = os.environ.get("HF_OFFLINE", "").lower() in ("1", "true", "yes")
        self.local_files_only = (offline if local_files_only is None
                                 else local_files_only)
        print(f"[Embedder] loading {self.model_name} on {DEVICE} "
              f"(local_files_only={self.local_files_only}) ...")
        self.model = SentenceTransformer(
            self.model_name, device=DEVICE,
            local_files_only=self.local_files_only,
        )
        print("[Embedder] ready.")

    def encode(self, texts, batch_size: int | None = None, show_progress: bool = False) -> np.ndarray:
        """批量编码并归一化（cosine 可直接用内积）。"""
        return self.model.encode(
            texts,
            batch_size=batch_size or EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]
