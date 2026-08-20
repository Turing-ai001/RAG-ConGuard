"""P1 步骤3：LLM 兜底路径本地验证（不上服务器，纯 mock）。

验证目标（对照步骤3要求）：
    1. 兜底分支正确触发 —— 仅低置信（NLI max score < 0.7）pair 进入兜底，
       高置信 pair 保持 NLI 结果（provenance="nli"）
    2. 融合公式按预期计算 —— score = alpha*s_NLI + (1-alpha)*s_LLM
       （alpha = NLI_LLM_FUSION_ALPHA = 0.3，极性/类型以 LLM 为准）
    3. 最终边写入图结构 —— Relation 的 src/dst/type/score/provenance/polarity
       正确进入 ClaimGraph 的 pos/neg/rel 边集合

手段：假 NLI（FakeNLIScorer，模拟低置信）+ mock LLM（按 prompt 内容返回预设
JSON，含解析容错 case）。不加载任何真实模型，逻辑链路确定性可断言。

用法：
    python scripts/_verify_fallback.py
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from config import NLI_CONFIDENCE_THRESHOLD, NLI_LLM_FUSION_ALPHA
from graph import ClaimGraph
from relation.extractor import RelationExtractor
from relation.llm_judge import LLMJudge
from relation.nli import numeric_conflict_check
from relation.types import RelationType

N_LABELS = 3   # contradiction / entailment / neutral


class FakeNLIScorer:
    """模拟 NLIScorer.classify 的行为契约：
        - 数值启发式预检（与真实 classify 一致）：命中 → (CONTRADICT, 1.0)
        - 指定高置信对：不触发兜底
        - 其余：低置信 unrelated(0.55) → 触发 LLM 兜底
    """

    def __init__(self, high_pairs: dict[tuple[str, str], tuple[RelationType, float]] = None):
        self.high = high_pairs or {}

    def classify(self, premises, hypotheses, threshold=NLI_CONFIDENCE_THRESHOLD):
        out = []
        for p, h in zip(premises, hypotheses):
            key = (p, h)
            if numeric_conflict_check(p, h) is not None:
                t, conf = RelationType.CONTRADICT, 1.0     # 数值启发式拦截
            elif key in self.high:
                t, conf = self.high[key]                   # 高置信 NLI 结果
            else:
                t, conf = RelationType.UNRELATED, 0.55     # 低置信，触发兜底
            probs = np.zeros(N_LABELS, dtype=np.float64)
            probs[0] = conf
            out.append((t, conf, probs))
        return out


class MockLLM:
    """mock LLM 后端：按 (src_mark, dst_mark) 组合精确匹配预设响应。
    注意：prompt 同时含 Claim A/B 两个标记，按词匹配会误中 src 标记，
    因此按组合 key 匹配（测试自身的确定性要求）。"""

    def __init__(self, pair_responses: dict[tuple[str, str], str]):
        self.responses = pair_responses    # key = (src_mark, dst_mark)
        self.calls: list[str] = []         # 记录收到的 prompt（验证触发范围）

    def complete(self, prompts, max_new_tokens=512):
        self.calls.extend(prompts)
        out = []
        for pr in prompts:
            marks = re.findall(r"\[([a-z]+)\]", pr)
            key = (marks[0], marks[1]) if len(marks) >= 2 else None
            out.append(self.responses.get(
                key, '{"relation": "unrelated", "confidence": 0.5}'))
        return out


def _mk_claims():
    """5 个 claim：两两构成 20 个有向对。标记词决定 mock LLM 的返回。"""
    return [
        {"claim_id": "c1", "text": "the series has [apple] 24 episodes",
         "meta": {"poisoned": True}},
        {"claim_id": "c2", "text": "the series has [banana] 23 episodes",
         "meta": {"poisoned": True}},
        {"claim_id": "c3", "text": "the [cherry] song was recorded by artist X",
         "meta": {"poisoned": False}},
        {"claim_id": "c4", "text": "the [date] concert happened in 2024",
         "meta": {"poisoned": False}},
        {"claim_id": "c5", "text": "the [elderberry] film won awards",
         "meta": {"poisoned": False}},
    ]


def main():
    passed = 0
    total = 0

    def check(name, cond, detail=""):
        nonlocal passed, total
        total += 1
        if cond:
            passed += 1
            print(f"  PASS {name}")
        else:
            print(f"  FAIL {name}: {detail}")

    # ========== A. 解析层（LLMJudge._parse 三层容错） ==========
    print("\n[A] LLMJudge 响应解析（三层容错）")
    judge = LLMJudge(llm=MockLLM({}))
    cases = [
        ('{"relation": "contradict", "confidence": 0.9}',
         RelationType.CONTRADICT, 0.9),
        ('```json\n{"relation": "entail", "confidence": 0.8}\n```',
         RelationType.ENTAIL, 0.8),            # 围栏格式
        ('I think it is support because...',
         RelationType.SUPPORT, 0.5),           # 无 JSON → 扫描关系词，默认 conf
        ('{"relation": "refute"}',
         RelationType.REFUTE, 0.5),            # 缺 confidence → 默认 0.5
        ('{"relation": "unrelated", "confidence": 1.0}',
         RelationType.UNRELATED, 1.0),
        ('{"relation": "banana", "confidence": 2.0}',
         RelationType.UNRELATED, 1.0),         # 非法关系→UNRELATED；conf 钳位到 1.0
    ]
    for resp, exp_t, exp_c in cases:
        t, c = judge._parse(resp)
        check(f"parse({resp[:40]}...) -> {exp_t.value}/{exp_c}",
              t is exp_t and abs(c - exp_c) < 1e-9,
              f"got {t}/{c}")

    # ========== A2. 步骤4.5：真实 LLM 输出变体鲁棒性 ==========
    print("\n[A2] 真实 LLM 输出变体（步骤4.5，5 个 case）")
    variants = [
        ("a) JSON 前后多余空白/换行",
         "\n\n{\"relation\": \"entail\", \"confidence\": 0.85}\n\n",
         RelationType.ENTAIL, 0.85),
        ("b) confidence 为字符串而非数字",
         "{\"relation\": \"contradict\", \"confidence\": \"0.9\"}",
         RelationType.CONTRADICT, 0.9),
        ("c) relation 大小写不一致",
         "{\"relation\": \"ENTAIL\", \"confidence\": 0.8}",
         RelationType.ENTAIL, 0.8),
        ("d) 多余字段（reasoning）",
         "{\"relation\": \"unrelated\", \"confidence\": 0.7, \"reasoning\": \"...\"}",
         RelationType.UNRELATED, 0.7),
        ("e) 嵌套 markdown 代码块 + 尾部文本",
         "```json\n{\"relation\": \"entail\", \"confidence\": 0.85}\n```\n以上是结果",
         RelationType.ENTAIL, 0.85),
    ]
    for name, resp, exp_t, exp_c in variants:
        t, c = judge._parse(resp)
        ok = t is exp_t and abs(c - exp_c) < 1e-9
        check(f"{name} -> {exp_t.value}/{exp_c}",
              ok, f"got {t.value}/{c} (resp={resp[:50]!r})")

    # ========== B. 触发 + 融合 + 图写入（端到端 extract_all） ==========
    print("\n[B] 兜底触发 / 融合公式 / 图写入")
    claims = _mk_claims()
    # 高置信对：c4->c5（无数字，不触发数值启发式）判定 entail(0.95)，不应走兜底
    high_pairs = {(claims[3]["text"], claims[4]["text"]): (RelationType.ENTAIL, 0.95)}
    llm_responses = {
        ("apple", "cherry"): '{"relation": "contradict", "confidence": 0.9}',      # c1->c3
        ("banana", "date"): '{"relation": "entail", "confidence": 0.8}',           # c2->c4
        ("cherry", "elderberry"): '{"relation": "support", "confidence": 0.7}',    # c3->c5
        ("date", "apple"): '{"relation": "refute", "confidence": 0.6}',            # c4->c1
        ("elderberry", "banana"): '{"relation": "unrelated", "confidence": 0.9}',  # c5->c2
    }
    nli = FakeNLIScorer(high_pairs)
    llm = MockLLM(llm_responses)
    extractor = RelationExtractor(nli, LLMJudge(llm))
    edges = extractor.extract_all(claims)

    # --- 1) 触发：20 对有向对 = 1 高置信 NLI + 2 数值启发式拦截(双向) + 17 低置信 LLM ---
    n_nli = sum(1 for e in edges if e.provenance == "nli")
    n_llm = sum(1 for e in edges if e.provenance == "llm")
    check("触发范围：20 对 = 3 NLI（1 高置信 + 2 数值拦截 c1<->c2）+ 17 LLM",
          n_nli == 3 and n_llm == 17, f"nli={n_nli} llm={n_llm}")
    check("兜底调用次数 = 17（mock LLM 收到 17 个 prompt）", len(llm.calls) == 17,
          f"calls={len(llm.calls)}")
    hi = next(e for e in edges if e.src == "c4" and e.dst == "c5")
    check("高置信对保持 NLI 结果（entail, 0.95, 无融合）",
          hi.provenance == "nli" and hi.type is RelationType.ENTAIL
          and abs(hi.score - 0.95) < 1e-9, f"got {hi}")
    # 数值启发式优先于 NLI：c1(24集) vs c2(23集) 被直接判矛盾 conf=1.0
    num = next(e for e in edges if e.src == "c1" and e.dst == "c2")
    check("数值启发式优先：c1->c2(24 vs 23) 拦截为 contradict/1.0",
          num.type is RelationType.CONTRADICT and abs(num.score - 1.0) < 1e-9,
          f"got {num}")

    # --- 2) 融合公式：score = 0.3*s_NLI + 0.7*s_LLM（s_NLI = 0.55） ---
    a = NLI_LLM_FUSION_ALPHA
    expect = {
        ("c1", "c3"): (RelationType.CONTRADICT, a * 0.55 + (1 - a) * 0.9),
        ("c2", "c4"): (RelationType.ENTAIL, a * 0.55 + (1 - a) * 0.8),
        ("c3", "c5"): (RelationType.SUPPORT, a * 0.55 + (1 - a) * 0.7),
        ("c4", "c1"): (RelationType.REFUTE, a * 0.55 + (1 - a) * 0.6),
        ("c5", "c2"): (RelationType.UNRELATED, a * 0.55 + (1 - a) * 0.9),
    }
    by_pair = {(e.src, e.dst): e for e in edges}
    for (s, d), (exp_t, exp_s) in expect.items():
        e = by_pair.get((s, d))
        check(f"融合边 {s}->{d} = {exp_t.value} @ {exp_s:.4f}",
              e is not None and e.type is exp_t and abs(e.score - exp_s) < 1e-9,
              f"got {e}")
        if e is not None:
            check(f"极性 {s}->{d} = {exp_t.polarity}（Relation.polarity）",
                  e.polarity == exp_t.polarity, f"got {e.polarity}")

    # --- 3) 图写入：ClaimGraph 的 pos/neg/rel 边集合与极性一致 ---
    g = ClaimGraph(claims, edges)
    n_pos = len(g.pos_edges)
    n_neg = len(g.neg_edges)
    n_rel = len(g.rel_edges)
    check("图写入：pos+neg = rel（正负边互斥且都算相关边）",
          n_pos + n_neg == n_rel, f"pos={n_pos} neg={n_neg} rel={n_rel}")
    # 类型→极性的预期计数：entail/support 正，contradict/refute 负，unrelated 不计
    type_counts = {}
    for e in edges:
        type_counts[e.type] = type_counts.get(e.type, 0) + 1
    exp_pos = type_counts.get(RelationType.ENTAIL, 0) + type_counts.get(RelationType.SUPPORT, 0)
    exp_neg = type_counts.get(RelationType.CONTRADICT, 0) + type_counts.get(RelationType.REFUTE, 0)
    check(f"图写入：pos_edges={n_pos} == 正极性类型数 {exp_pos}",
          n_pos == exp_pos, f"got {n_pos} vs {exp_pos}")
    check(f"图写入：neg_edges={n_neg} == 负极性类型数 {exp_neg}",
          n_neg == exp_neg, f"got {n_neg} vs {exp_neg}")
    # 所有 LLM 边 provenance/score 已写入（抽查）
    sample = next(e for e in edges if (e.src, e.dst) == ("c1", "c3"))
    check("边字段完整：type/score/provenance/polarity",
          sample.provenance == "llm"
          and sample.type is RelationType.CONTRADICT
          and sample.polarity == -1
          and 0 < sample.score < 1,
          f"got {sample}")

    print(f"\n{'=' * 60}")
    print(f"结果: {passed}/{total} 通过")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
