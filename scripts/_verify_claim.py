"""里程碑验证：真实 LLM 的 claim 提取端到端（临时脚本，验证后可删）。

链路：query → 多视角检索（BGE+FAISS+BM25+KG）→ 主张提取（真实 Qwen2.5-1.5B）
输出：每 query 的检索文档 + 解析出的 claims + RAG 生成回答。
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_OFFLINE", "1")   # 模型已缓存，离线加载
os.environ.setdefault("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")  # 本机验证用 1.5B；服务器 7B

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import HF_MODEL, TOP_K
from llm import get_backend
from pipeline import Pipeline
from retrieval import Embedder, KnowledgeBase
from retrieval.kb import load_poisoned_docs
from claim import ClaimExtractor

N_QUERIES = 2

# 真实生成模型（本地 1.5B 验证；服务器换 Qwen2.5-7B 只需改 HF_MODEL / 换 vllm 后端）
llm = get_backend("hf")
print(f"[verify] LLM: {HF_MODEL} (backend={llm.name})")

embedder = Embedder()
kb = KnowledgeBase.load("smoke_kb", embedder)
extractor = ClaimExtractor(llm)
pipe = Pipeline(kb, extractor, llm)

poisoned = load_poisoned_docs("nq")
print("=" * 70)
print("里程碑验证：query → 检索文档 + 主张列表 + 生成")
print("=" * 70)

ok = True
for qid, item in list(poisoned.items())[:N_QUERIES]:
    q = item["question"]
    print(f"\n--- [{qid}] query: {q}")
    print(f"    正确答案: {item['correct answer']!r} | 投毒答案: {item['incorrect answer']!r}")

    out = pipe.run(q, extract_claims=True)
    print(f"    检索文档 {len(out['docs'])} 条:")
    for d in out["docs"]:
        src = "poisoned" if "poisoned" in d["doc_id"] else "corpus"
        print(f"      [{src}] {d['text'][:90]}...")

    print(f"    主张列表（真实 LLM 提取）:")
    for c in out["claims"]:
        for x in c["claims"]:
            print(f"      · rel={x['relevance']:.2f} {x['text'][:90]}")
        if not c["claims"]:
            print(f"      (文档 {c['doc_id']} 无 claims)")
            ok = False

    print(f"    生成回答: {out['answer'][:80]!r}")
    print()

print("=" * 70)
print("结论:", "✅ 里程碑达成：query → 检索文档 + 主张列表 真实 LLM 端到端跑通"
      if ok else "⚠️ 存在空 claims，检查 prompt/解析")
print("=" * 70)
