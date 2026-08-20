"""P0 端到端冒烟测试（本地 CPU + Mock LLM）。

验证链路：多视角检索(FAISS+BM25+融合) → 主张提取(JSON 解析容错) → 生成。
全部确定性输出，可重复运行。

用法（在项目根目录）：
    python tests/smoke_test.py [--dataset nq] [--queries 5]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 本机验证默认离线加载模型（模型必已就位，避免 hf_hub 联网检查在无外网时超时）
# 必须早于任何 huggingface/transformers import；需联网下载时设 HF_OFFLINE=0
os.environ.setdefault("HF_OFFLINE", "1")

# Windows 控制台默认 GBK，输出 emoji 会崩；统一 UTF-8（错误用占位符兜底）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import TOP_K
from claim import ClaimExtractor
from llm import get_backend
from pipeline import Pipeline
from retrieval import Embedder, build_mini_kb
from retrieval.kb import load_poisoned_docs

_SENT = re.compile(r"(?<=[.!?])\s+")


def mock_llm_handler(prompt: str) -> str:
    """Mock 响应：claim 提取 prompt 返回合法 JSON；其余返回固定串。"""
    if "Document:" in prompt:
        doc = prompt.split("Document:", 1)[1].strip()
        sentences = [s for s in _SENT.split(doc) if s.strip()][:2]
        claims = [{"text": s, "relevance": 0.8} for s in sentences]
        return json.dumps({"claims": claims})
    return "MOCK ANSWER"


def main(dataset: str, n_queries: int):
    print("=" * 60)
    print(f"P0 冒烟测试 | dataset={dataset} | queries={n_queries} | top_k={TOP_K}")
    print("=" * 60)

    # 1) Embedder（BGE-base，CPU；首次运行会下载模型 ~440MB）
    embedder = Embedder()

    # 2) 迷你 KB：前 120 条投毒文档（够验证检索/融合，CPU 编码快）
    kb = build_mini_kb(embedder, dataset, max_docs=120)
    print(f"[smoke] KB docs={len(kb.entries)}")

    # 3) 检索：n 个目标 query，验证"该 query 自己的投毒文档"能被多视角检索命中
    poisoned = load_poisoned_docs(dataset)
    queries = list(poisoned.items())[:n_queries]
    hit = 0
    for qid, item in queries:
        q = item["question"]
        docs = kb.search(q, top_k=TOP_K)
        own_poisoned = [d for d in docs if f":{qid}:" in d["doc_id"]]
        if own_poisoned:
            hit += 1
        q_ents = kb.kg.extract_query_entities(q)
        view_hits = {
            "semantic": sum(1 for d, _ in kb.vector_store.search(embedder.encode_query(q), TOP_K)
                            if f":{qid}:" in d),
            "keyword": sum(1 for d, _ in kb.bm25.search(q, TOP_K) if f":{qid}:" in d),
            "kg": sum(1 for d, _ in kb.kg.search(q_ents, TOP_K) if f":{qid}:" in d),
        }
        print(f"  [{qid}] {q[:60]}...")
        print(f"       top-k 自身投毒={len(own_poisoned)}/{TOP_K} | 视角命中={view_hits} | "
              f"query 实体={q_ents[:4]}")
    print(f"[smoke] 自身投毒命中率 {hit}/{len(queries)}")
    assert hit >= max(1, len(queries) // 2), "检索命中率过低，多视角检索有问题"

    # 4) KB 持久化往返
    kb.save("smoke_kb")
    kb2 = KnowledgeBase.load("smoke_kb", embedder)
    assert len(kb2.entries) == len(kb.entries), "KB save/load 不一致"

    # 5) 主张提取 + 生成（Mock LLM）
    llm = get_backend("mock")
    llm.handler = mock_llm_handler
    extractor = ClaimExtractor(llm)
    pipe = Pipeline(kb2, extractor, llm)

    q, item = queries[0]
    out = pipe.run(item["question"], extract_claims=True)
    assert out["claims"] and len(out["claims"]) > 0, "claims 为空"
    c0 = out["claims"][0]
    assert "doc_id" in c0 and "claims" in c0 and c0["claims"], "claims 结构异常"
    print(f"[smoke] 主张提取 OK: {len(out['claims'])} 文档各 {[len(c['claims']) for c in out['claims']]} 条")
    print(f"[smoke] 生成回答: {out['answer']!r}")
    assert out["answer"] == "MOCK ANSWER"

    # 6) 批量接口
    answers = llm.complete([q, "q2"])
    assert isinstance(answers, list) and len(answers) == 2
    print("[smoke] LLM 批量接口 OK")

    print("\n✅ P0 冒烟测试全部通过：检索 / 融合 / 持久化 / 主张提取 / 生成 / 批量")


if __name__ == "__main__":
    from retrieval.kb import KnowledgeBase  # noqa: F401 (KB save/load 用)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq")
    ap.add_argument("--queries", type=int, default=5)
    args = ap.parse_args()
    main(args.dataset, args.queries)
