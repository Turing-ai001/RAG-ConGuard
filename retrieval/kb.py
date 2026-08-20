"""知识库（KB）组装与多视角检索入口。

KB = 原始语料 + 投毒文档，统一 doc_id。每个 KB 持久化到 data/index/{name}/：
    docs.jsonl     文档本体（doc_id, text, meta）
    index.faiss    语义向量索引（BGE）
    ids.json       向量索引的 doc_id 映射
    bm25.pkl       BM25 索引
"""
import json
import pickle
from pathlib import Path

from config import INDEX_DIR, POISONED_DIR, TOP_K
from retrieval.fusion import rrf_fuse
from retrieval.kg import EntityKGStore, KGStore, default_kg_store
from retrieval.keyword import BM25Index
from retrieval.vector_store import VectorStore


# ================= 投毒文档加载（复用旧项目 PoisonedRAG 数据） =================
def load_poisoned_docs(dataset: str = "nq") -> dict:
    """加载 data/poisoned/{dataset}.json。
    格式（与旧项目一致）：{query_id: {id, question, "correct answer", "incorrect answer", adv_texts: [...]}}
    """
    path = POISONED_DIR / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(f"poisoned data not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def poisoned_docs_to_entries(dataset: str) -> list[dict]:
    """把投毒文档转成 KB doc 条目，meta 标注 poisoned 归属（攻击/目标 query）。"""
    data = load_poisoned_docs(dataset)
    entries = []
    for qid, item in data.items():
        for adv in item["adv_texts"]:
            entries.append({
                "doc_id": f"{dataset}:poisoned:{qid}:{len(entries)}",
                "text": adv,
                "meta": {
                    "source": "poisoned",
                    "dataset": dataset,
                    "target_query_id": qid,
                    "target_question": item.get("question", ""),
                    "target_answer": item.get("incorrect answer", ""),
                },
            })
    return entries


# ================= 知识库 =================
class KnowledgeBase:
    """多视角检索的单一入口：semantic(FAISS) + keyword(BM25) + kg(接口)。"""

    def __init__(self, name: str, doc_entries: list[dict], embedder, kg_store: KGStore | None = None):
        self.name = name
        self.entries = doc_entries
        self.doc_id2entry = {e["doc_id"]: e for e in doc_entries}
        self.embedder = embedder
        self.kg = kg_store or default_kg_store()

        doc_ids = [e["doc_id"] for e in doc_entries]
        texts = [e["text"] for e in doc_entries]
        self.vector_store = VectorStore(embedder.encode([""]).shape[1])
        self.vector_store.add(doc_ids, embedder.encode(texts, show_progress=True))
        self.bm25 = BM25Index(doc_ids, texts)
        # KG 视角：实体倒排 + 规则三元组（语料 title 字段参与建索引）
        titles = [str(e.get("meta", {}).get("title", "")) for e in doc_entries]
        self.kg = kg_store or EntityKGStore(doc_ids, texts, titles)

    # ---------- 多视角检索 ----------
    def search(self, query: str, top_k: int = TOP_K, query_entities: list[str] | None = None) -> list[dict]:
        """多视角检索 + RRF 融合。返回 [{doc_id, text, meta, rrf_score}, ...]。"""
        q_emb = self.embedder.encode_query(query)
        q_ents = self.kg.extract_query_entities(query) if query_entities is None else query_entities
        view_results = {
            "semantic": self.vector_store.search(q_emb, top_k * 4),
            "keyword": self.bm25.search(query, top_k * 4),
            "kg": self.kg.search(q_ents, top_k * 4),
        }
        fused = rrf_fuse(view_results, top_k=top_k)
        return [
            {**self.doc_id2entry[doc_id], "rrf_score": score}
            for doc_id, score in fused
        ]

    # ---------- 持久化 ----------
    def save(self, name: str | None = None) -> Path:
        root = INDEX_DIR / (name or self.name)
        root.mkdir(parents=True, exist_ok=True)
        (root / "docs.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in self.entries), encoding="utf-8"
        )
        self.vector_store.save(root / "index.faiss", root / "ids.json")
        (root / "bm25.pkl").write_bytes(pickle.dumps(self.bm25))
        if hasattr(self.kg, "index"):   # EntityKGStore 才持久化；EmptyKG 等无状态实现跳过
            (root / "kg.pkl").write_bytes(pickle.dumps(self.kg))
        print(f"[KB] saved to {root}")
        return root

    @classmethod
    def load(cls, name: str, embedder) -> "KnowledgeBase":
        root = INDEX_DIR / name
        entries = [
            json.loads(line)
            for line in (root / "docs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        kb = cls.__new__(cls)
        kb.name = name
        kb.entries = entries
        kb.doc_id2entry = {e["doc_id"]: e for e in entries}
        kb.embedder = embedder
        kg_path = root / "kg.pkl"
        kb.kg = (pickle.loads(kg_path.read_bytes()) if kg_path.exists() else default_kg_store())
        kb.vector_store = VectorStore.load(root / "index.faiss", root / "ids.json")
        kb.bm25 = pickle.loads((root / "bm25.pkl").read_bytes())
        print(f"[KB] loaded {name}: {len(entries)} docs")
        return kb


def build_mini_kb(embedder, dataset: str = "nq", max_docs: int | None = None) -> KnowledgeBase:
    """P0 冒烟用迷你 KB：仅投毒文档（可选截断，控制 CPU 编码耗时）。
    正式语料构建见 scripts/download_datasets.py + scripts/build_kb.py。
    """
    entries = poisoned_docs_to_entries(dataset)
    if max_docs is not None:
        entries = entries[:max_docs]
    print(f"[build_mini_kb] {len(entries)} poisoned docs for {dataset}")
    return KnowledgeBase(f"mini_{dataset}", entries, embedder)
