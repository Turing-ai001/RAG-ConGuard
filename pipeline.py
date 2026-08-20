"""RAG 主管线（P0 版）：检索 → 主张提取 → 生成。

接口对齐旧项目（RAG prompt 沿用，保证与攻击论文实验可比）；
claim 提取是新论文的核心新增环节，接在检索后、生成前。
"""
from config import RAG_PROMPT, TOP_K
from claim import ClaimExtractor
from llm.base import LLMBackend
from retrieval.kb import KnowledgeBase


class Pipeline:
    def __init__(self, kb: KnowledgeBase, extractor: ClaimExtractor, llm: LLMBackend):
        self.kb = kb
        self.extractor = extractor
        self.llm = llm

    def run(self, query: str, top_k: int = TOP_K,
            extract_claims: bool = True) -> dict:
        """单 query 全链路。返回：
        {answer, docs: [...], claims: [...] (可选)}
        """
        docs = self.kb.search(query, top_k=top_k)

        claims = None
        if extract_claims:
            claims = self.extractor.extract(query, docs)

        context_str = "\n".join(d["text"] for d in docs)
        prompt = RAG_PROMPT.replace("[question]", query).replace("[context]", context_str)
        answer = self.llm.complete(prompt)

        return {"answer": answer, "docs": docs, "claims": claims}

    def run_batch(self, queries: list[str], **kw) -> list[dict]:
        return [self.run(q, **kw) for q in queries]
