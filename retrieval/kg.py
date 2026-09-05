"""KG 三元组检索视角。

P0 目标：多视角检索中的"KG 视角"真实可用，且不依赖外部图谱服务。
实现方案（轻量、稳定、可替换）：
    1. 实体抽取（规则版）：大写词序列 / 引号实体 / 数字年份，
       对 query 与文档通用（extract_entities）。
    2. 实体倒排索引（EntityIndex）：entity -> [(doc_id, tf)]，
       检索打分 = Σ_entity IDF(entity) × (1 + log tf)，等价于轻量 TF-IDF 实体检索。
    3. 规则三元组抽取（RuleBasedTripleExtractor）：从文本抽 (head, rel, tail)，
       只抽高置信模式（宁少勿错，避免错误三元组污染检索）。
       三元组中的 head/tail 一并计入实体倒排，使三元组检索退化为实体检索的特例。

后续接入真实图谱（如 Wikidata 子集 / LLM 抽取）只需：
    实现 KGStore 协议（search/extract_query_entities），并在 default_kg_store 替换。
"""
from __future__ import annotations

import math
import multiprocessing as mp
import re
from collections import defaultdict
from typing import Protocol

_N_WORKERS = 16   # 全量构建：per-doc 实体抽取是 7 个正则扫描，并行化

# ================= 实体抽取（规则版） =================

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with",
    "from", "by", "as", "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "his", "her", "their", "they", "he", "she",
    "not", "no", "do", "does", "did", "has", "have", "had", "will", "would",
    "can", "could", "should", "which", "who", "whom", "when", "where", "what",
    "how", "than", "into", "during", "before", "after", "over", "under", "between",
}
# 大写词序列：潜在实体（专有名词），如 "Chicago Fire"；
# 段间允许 1 个小写专名前缀词（van/der/de/da...，如 "Vincent van Gogh"）。
_ENTITY_SEQ = r"[A-Z][a-zA-Z0-9]+(?:\s+(?:[a-z]{1,4}\s+)?[A-Z][a-zA-Z0-9'-]+)+"
_ENTITY_SEQ_RE = re.compile(r"\b(" + _ENTITY_SEQ + r")\b")
# 单大写词实体（Paris / London 等），排除句首词（可能是普通名词句首大写）。
_ENTITY_WORD_RE = re.compile(r"\b([A-Z][a-z]{1,20})\b")
_QUOTED_RE = re.compile(r'"([^"]{2,60})"')
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_LEAD_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")

_SENT_BOUNDARY = set(".!?\n")


def _query_words(query: str) -> list[str]:
    """query 的非停用词单词（去重保序），供伪实体兜底用。"""
    from retrieval.keyword import tokenize
    return list(dict.fromkeys(w for w in tokenize(query) if w not in _STOPWORDS))


def _normalize_entity(e: str) -> str:
    """小写 / strip / 去前导冠词（"The Louvre" -> "louvre"），与裸实体名对齐。"""
    e = _LEAD_ARTICLE_RE.sub("", e.strip().lower())
    return e.rstrip(".,;")


def extract_entities(text: str, max_per_doc: int = 50) -> list[str]:
    """从文本抽取候选实体，小写化、去停用词、按出现频率截断。

    覆盖：大写词序列（如 "Chicago Fire"）、单大写词（如 "Paris"）、
    引号实体（如 "The Scream"）、独立年份（如 "1945"）。
    句首词排除（多为普通名词句首大写）。不保证全对 —— 检索视角只需宽召回，
    错误实体由 IDF 与 RRF 融合兜底。
    """
    if not text:
        return []
    freq: dict[str, int] = defaultdict(int)
    seen: set[str] = set()

    def _add(e: str):
        e = _normalize_entity(e)
        if len(e) < 2 or len(e) > 40:
            return
        words = e.split()
        if all(w in _STOPWORDS for w in words):
            return
        if e in seen:
            freq[e] += 1
            return
        seen.add(e)
        freq[e] = 1

    def _skip_sentence_start(m) -> bool:
        return m.start() == 0 or text[m.start() - 1] in _SENT_BOUNDARY

    for m in _ENTITY_SEQ_RE.finditer(text):
        _add(m.group(1))          # 多词序列：句首 "The Louvre" 仍是强实体（去冠词后入库）
    for m in _ENTITY_WORD_RE.finditer(text):
        if not _skip_sentence_start(m):
            _add(m.group(1))      # 单大写词：句首多为普通名词句首大写，排除
    for m in _QUOTED_RE.finditer(text):
        _add(m.group(1))
    for m in _YEAR_RE.finditer(text):
        _add(m.group(1))

    if not freq:
        return []
    ranked = sorted(freq.items(), key=lambda kv: -kv[1])
    return [e for e, _ in ranked[:max_per_doc]]


# ================= 规则三元组抽取 =================

# (head, rel, tail) 高置信模式。head 用大写序列（与实体抽取同款），tail 宽容（含普通名词）。
_TRIPLE_PATTERNS = [
    # "X is a Y" / "X was an Y"
    re.compile(rf"({_ENTITY_SEQ})\s+(?:is|was|are|were)\s+(?:a|an|the)\s+([a-zA-Z][a-zA-Z0-9' -]{{1,40}})", re.I),
    # "X is located in Y" / "was founded in Y" / "was born in Y"
    re.compile(rf"({_ENTITY_SEQ})\s+(?:is|was)\s+(?:located|situated|founded|established|built|born|released|published)\s+in\s+([A-Za-z][A-Za-z0-9' -]{{1,40}})", re.I),
    # "X is the capital of Y"
    re.compile(rf"({_ENTITY_SEQ})\s+is\s+the\s+capital\s+of\s+([A-Za-z][A-Za-z0-9' -]{{1,40}})", re.I),
]


def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """规则抽取 (head, relation, tail)。关系取模式名，宁少勿错。"""
    if not text:
        return []
    out: list[tuple[str, str, str]] = []
    for rel, pat in (("is_a", _TRIPLE_PATTERNS[0]),
                     ("located_in", _TRIPLE_PATTERNS[1]),
                     ("capital_of", _TRIPLE_PATTERNS[2])):
        for m in pat.finditer(text):
            head = _normalize_entity(m.group(1))
            tail = _normalize_entity(m.group(2))
            if head and tail and len(head) <= 40:
                out.append((head, rel, tail))
    return out


# ================= 实体倒排索引 =================

class EntityIndex:
    """entity -> [(doc_id, tf)] 倒排 + 文档 IDF。"""

    def __init__(self):
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.df: dict[str, int] = {}
        self.n_docs = 0

    def add_doc(self, doc_id: str, entities: list[str]) -> None:
        """写入一个文档的实体集合（含三元组 head/tail，rel 前加 'rel:' 前缀区分）。"""
        counts: dict[str, int] = defaultdict(int)
        for e in entities:
            counts[e] += 1
        self.n_docs += 1
        for e, c in counts.items():
            self.postings[e].append((doc_id, c))
            self.df[e] = self.df.get(e, 0) + 1   # df 增量维护，O(1)/实体

    def idf(self, entity: str) -> float:
        df = self.df.get(entity, 0)
        if df == 0:
            return 0.0
        return math.log((self.n_docs + 1) / (df + 0.5)) + 1.0

    def search(self, query_entities: list[str], k: int) -> list[tuple[str, float]]:
        """打分：score(doc) = Σ_e IDF(e) × (1 + log tf(e, doc))。"""
        if not query_entities:
            return []
        acc: dict[str, float] = defaultdict(float)
        for e in query_entities:
            w = self.idf(e)
            if w == 0:
                continue
            for doc_id, tf in self.postings.get(e, []):
                acc[doc_id] += w * (1.0 + math.log(tf))
        ranked = sorted(acc.items(), key=lambda kv: -kv[1])
        return ranked[:k]


# ================= KGStore 协议与实现 =================

class KGStore(Protocol):
    """KG 三元组检索协议。

    search(query_entities, k) -> [(doc_id, score), ...] 降序。
    extract_query_entities(query) -> list[str]：从 query 抽取实体。
    """

    def search(self, query_entities: list[str], k: int) -> list[tuple[str, float]]:
        ...

    def extract_query_entities(self, query: str) -> list[str]:
        ...


def _extract_doc_worker(text: str, title: str):
    """单文档抽取（进程池 worker 用，无状态、可 pickle）：
    实体（title 权重 ×2）+ 三元组（head/tail/rel 一并入实体列表）。"""
    ents = extract_entities(text)
    if title:
        title_ents = extract_entities(title)
        ents = ents + title_ents + title_ents         # 标题实体出现两次 = 权重 ×2
    triples = extract_triples(text)
    for h, rel, t in triples:
        ents.append(h); ents.append(t); ents.append(f"rel:{rel}")
    return ents, triples


class EntityKGStore:
    """实体倒排 + 规则三元组的 KG 视角实现（不依赖外部图谱，可独立持久化）。"""

    name = "entity_kg"

    def __init__(self, doc_ids: list[str] | None = None,
                 texts: list[str] | None = None,
                 titles: list[str] | None = None):
        self.index = EntityIndex()
        self.triples: list[tuple[str, str, str]] = []
        if doc_ids is not None:
            self.build(doc_ids, texts or [], titles or [])

    # ---------- 构建 ----------
    def build(self, doc_ids: list[str], texts: list[str],
              titles: list[str] | None = None) -> None:
        """doc_ids/texts/titles 对齐；title 实体权重翻倍（标题实体更可靠）。"""
        titles = titles or [""] * len(doc_ids)
        # 全量 60 万文档：multiprocessing.starmap 在超大量 IPC 队列下实测死锁
        # （无报错、CPU 时间停增）；串行版本可靠，代价是构建慢几分钟。
        # 每 50000 条打点进度，可观测。
        for i, (doc_id, text, title) in enumerate(zip(doc_ids, texts, titles)):
            ents, triples = _extract_doc_worker(text, title)
            self.index.add_doc(doc_id, ents)
            self.triples.extend(triples)
            if (i + 1) % 50000 == 0:
                print(f"[KG] {i + 1}/{len(doc_ids)} docs indexed", flush=True)

    # ---------- 检索 ----------
    def extract_query_entities(self, query: str) -> list[str]:
        """query 实体：规则实体（大写序列等）+ 非停用词伪实体兜底。

        规则实体只覆盖大写/引号/年份 —— 全小写 query（如 NQ/MSMARCO）会抽空。
        伪实体（非停用词单词 + bigram）是"弱实体"：文档索引里没有对应实体时
        IDF=0 自动忽略，不产生噪声；命中索引实体时提供与 BM25 互补的 IDF×tf 打分。
        """
        ents = extract_entities(query)
        words = _query_words(query)
        seen = set(ents)
        for w in words:
            if w not in seen:
                seen.add(w)
                ents.append(w)
        for i in range(len(words) - 1):
            bg = f"{words[i]} {words[i + 1]}"
            if bg not in seen:
                seen.add(bg)
                ents.append(bg)
        return ents

    def search(self, query_entities: list[str], k: int) -> list[tuple[str, float]]:
        return self.index.search(query_entities, k)

    def stats(self) -> dict:
        return {
            "name": self.name,
            "entities": len(self.index.postings),
            "docs": self.index.n_docs,
            "triples": len(self.triples),
        }


class EmptyKG:
    """空实现：不返回任何结果。无 KG 数据时保证多视角检索仍正常。"""

    name = "empty"

    def search(self, query_entities: list[str], k: int) -> list[tuple[str, float]]:
        return []

    def extract_query_entities(self, query: str) -> list[str]:
        return []


def default_kg_store() -> KGStore:
    """无参空实例（供 KB.load / 快速冒烟用）；正式 KB 在 __init__ 时注入 EntityKGStore。"""
    return EmptyKG()
