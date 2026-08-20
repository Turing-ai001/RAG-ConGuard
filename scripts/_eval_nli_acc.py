"""P1 步骤2 评估：数字归一化启发式接入后的 126 样本重评估。

构造标签（与旧评估一致）：同组=蕴含（投毒↔投毒/正常↔正常），跨组=矛盾。

报告：
    a) 数值冲突子集（期望 contradict 且含数值差异）：启发式召回率（目标 >=95%）
       + 误伤数（目标 =0）
    b) 非数值冲突子集（期望 entail）：NLI 准确率（辅助指标 MILESTONE_NLI_ACC_NON_NUMERIC >=0.80）
    c) 整体：有符号图联盟纯度（主指标，对比启发式接入前）+ 相关边数变化
    d) 典型 case：一个正确拦截 / 一个正确放行 / 一个边界 case

用法：
    python scripts/_eval_nli_acc.py --dataset nq --num_queries 3
"""
import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import MILESTONE_NLI_ACC_NON_NUMERIC, MILESTONE_SIGNED_PURITY, TOP_K
from graph import ClaimGraph
from relation import NLIScorer
from relation.nli import numeric_conflict_check
from relation.types import Relation, RelationType
from retrieval import Embedder
from retrieval.kb import load_poisoned_docs

classes = ("entail", "contradict", "unrelated")


def doc_id_to_claim(doc_id: str) -> str:
    parts = doc_id.split(":")
    return parts[-2][:8] if len(parts) >= 3 else doc_id[:8]


def synthetic_normal_claims(item: dict, n: int) -> list[dict]:
    q = item["question"].rstrip("?").lower()
    ans = item["correct answer"]
    template = (
        "According to the official records, the answer to \"{q}\" is {ans}, "
        "and this is confirmed by multiple independent sources."
    )
    return [
        {"claim_id": f"normal-{i}", "text": template.format(q=q[:60], ans=ans)[:220],
         "meta": {"poisoned": False}}
        for i in range(n)
    ]


def expected_value(meta_a: dict, meta_b: dict) -> str:
    if meta_a.get("poisoned") == meta_b.get("poisoned"):
        return "entail"
    return "contradict"


def main(dataset: str, num_queries: int, synth_normal: int):
    nli = NLIScorer()
    embedder = Embedder()
    kb = __import__("retrieval.kb", fromlist=["build_mini_kb"]).build_mini_kb(embedder, dataset)
    poisoned = load_poisoned_docs(dataset)

    # ---------- 收集逐 pair 判定 ----------
    from relation.nli import _extract_numbers as _en
    pairs = []          # {exp, hit, num_diff, nli_raw, nli_pipe, src_id, dst_id, src, dst, detail}
    graphs_off = []     # 启发式关闭时的 ClaimGraph（对照）
    graphs_on = []      # 启发式开启时的 ClaimGraph
    num_related_off = num_related_on = 0

    for qid, item in list(poisoned.items())[:num_queries]:
        docs = kb.search(item["question"], top_k=TOP_K)
        claims = []
        for d in docs:
            poisoned_flag = d["doc_id"].startswith(f"{dataset}:poisoned")
            cid = f"{doc_id_to_claim(d['doc_id'])}-{len(claims)}"
            claims.append({"claim_id": cid, "text": d["text"][:220],
                           "meta": {"poisoned": poisoned_flag}})
        claims += synthetic_normal_claims(item, synth_normal)
        n = len(claims)

        # 诊断：检索到的 doc 构成（doc_id 是否本 query、是否含数字）
        diag_nums = {d["doc_id"].split(":")[-2][:8] + "-" + str(i): 0
                     for i, d in enumerate(docs)}
        hit_this_q = 0

        rels_off, rels_on = [], []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                src_c, dst_c = claims[i], claims[j]
                exp = expected_value(src_c["meta"], dst_c["meta"])
                src, dst = src_c["text"], dst_c["text"]

                hit = numeric_conflict_check(src, dst)
                # 含数值差异：双方候选数字非空且存在不同数值（数值冲突子集口径）
                _ca, _cb = _en(src), _en(dst)
                num_diff = bool(_ca and _cb
                                and any(a["value"] != b["value"]
                                        for a in _ca for b in _cb))
                t, conf, _ = nli.classify([src], [dst])[0]
                pipe = t.value if conf >= 0.7 else "unrelated"
                probs = nli.score_pairs([src], [dst])[0]
                raw = nli.labels[int(probs.argmax())]
                raw = {"entailment": "entail", "contradiction": "contradict",
                       "neutral": "unrelated"}.get(raw, "unrelated")

                pairs.append({"exp": exp, "hit": hit is not None, "num_diff": num_diff,
                              "nli_raw": raw, "nli_pipe": pipe,
                              "src": src_c["claim_id"], "dst": dst_c["claim_id"],
                              "src_text": src, "dst_text": dst,
                              "detail": hit[1] if hit else None})

                # 启发式关闭：纯 NLI 管线
                t_off = RelationType(pipe)
                rels_off.append(Relation(src_c["claim_id"], dst_c["claim_id"],
                                         t_off, conf if conf >= 0.7 else 0.0,
                                         provenance="nli"))
                # 启发式开启：命中 → 直接矛盾边 conf=1.0
                if hit:
                    rels_on.append(Relation(src_c["claim_id"], dst_c["claim_id"],
                                            RelationType.CONTRADICT, 1.0,
                                            provenance="numeric"))
                else:
                    rels_on.append(Relation(src_c["claim_id"], dst_c["claim_id"],
                                            t_off, conf if conf >= 0.7 else 0.0,
                                            provenance="nli"))
                if exp == "contradict" and hit:
                    hit_this_q += 1
        # 诊断输出：本 query 检索到的 docs（是否含数字、是否跨组命中）
        from relation.nli import _extract_numbers as _en
        _doc_diag = []
        for i, d in enumerate(docs):
            _nums = [f"{x['value']:g}{x['unit'] or ''}" for x in _en(d["text"])]
            _tag = "SELF" if d["doc_id"].startswith(f"{dataset}:poisoned") else "OTHER"
            _doc_diag.append(f"{_tag}:{d['doc_id'].split(':')[-2][:8]} nums={_nums}")
        print(f"  [diag {qid}] docs={_doc_diag} | 跨组命中 {hit_this_q}/20")
        graphs_off.append(ClaimGraph(claims, rels_off))
        graphs_on.append(ClaimGraph(claims, rels_on))

    # ---------- a) 数值冲突子集：召回率 + 误伤 ----------
    # 口径：期望 contradict 且双方含数值差异（启发式只对该子集负责）
    should_intercept = [p for p in pairs if p["exp"] == "contradict" and p["num_diff"]]
    hit_n = sum(1 for p in should_intercept if p["hit"])
    fp = [p for p in pairs if not (p["exp"] == "contradict" and p["num_diff"]) and p["hit"]]
    print("=" * 72)
    print(f"a) 数值冲突子集（期望 contradict 且含数值差异，{len(should_intercept)} 样本）")
    print("=" * 72)
    print(f"   启发式召回率: {hit_n / len(should_intercept):.1%}  ({hit_n}/{len(should_intercept)})"
          f"  [目标 >=95%]")
    print(f"   误伤数: {len(fp)}  [目标 =0]")
    for p in fp[:5]:
        print(f"     !! 误伤: {p['src']} -> {p['dst']} ({p['detail']})")

    # ---------- b) 非数值冲突子集：NLI 准确率 ----------
    # 口径：非数值冲突子集 = 全部 pair 减去数值冲突子集（含无数值 contradict 对）
    subset = [p for p in pairs if not (p["exp"] == "contradict" and p["num_diff"])]
    acc_raw = sum(1 for p in subset if p["nli_raw"] == p["exp"]) / len(subset)
    acc_pipe = sum(1 for p in subset if p["nli_pipe"] == p["exp"]) / len(subset)
    cm = Counter((p["exp"], p["nli_raw"]) for p in subset)
    print("\n" + "=" * 72)
    print(f"b) 非数值冲突子集（{len(subset)} 样本 = 66 entail + 40 无数值 contradict）—— 辅助指标")
    print("=" * 72)
    print(f"   NLI 原始 argmax 准确率: {acc_raw:.1%}  [目标 >= {MILESTONE_NLI_ACC_NON_NUMERIC:.0%}]"
          f"  {'达标' if acc_raw >= MILESTONE_NLI_ACC_NON_NUMERIC else '未达标'}")
    print(f"   NLI 管线(0.7阈值)准确率: {acc_pipe:.1%}")
    for (e, p), v in sorted(cm.items()):
        print(f"     期望{e} 判为{p}: {v}")

    # ---------- c) 整体：联盟纯度 + 边数 ----------
    def gsum(graphs):
        n_coal = sum(g.summary("signed")["n_coalitions"] for g in graphs) / len(graphs)
        purity = sum(g.summary("signed")["avg_purity"] for g in graphs) / len(graphs)
        n_rel = sum(len(g.rel_edges) for g in graphs)
        return n_coal, purity, n_rel

    c_off, p_off, rel_off = gsum(graphs_off)
    c_on, p_on, rel_on = gsum(graphs_on)
    print("\n" + "=" * 72)
    print("c) 整体（每 query 平均）—— 主指标")
    print("=" * 72)
    print(f"   启发式接入前: 联盟 {c_off:.1f} | 纯度 {p_off:.3f} | 相关边 {rel_off} 条")
    print(f"   启发式接入后: 联盟 {c_on:.1f} | 纯度 {p_on:.3f} | 相关边 {rel_on} 条"
          f"  [主指标 >= {MILESTONE_SIGNED_PURITY:.2f}]"
          f"  {'达标' if p_on >= MILESTONE_SIGNED_PURITY else '未达标'}")
    print(f"   相关边变化: {rel_off} -> {rel_on}（{rel_on / max(rel_off, 1) - 1:+.0%}，"
          f"预期约 +50% 多召回数值冲突边）")

    # ---------- d) 典型 case ----------
    from relation.nli import _extract_numbers
    print("\n" + "=" * 72)
    print("d) 典型 case")
    print("=" * 72)
    hit_case = next(p for p in pairs if p["hit"])
    print(f"  [正确拦截] {hit_case['src']} -> {hit_case['dst']}")
    print(f"      A: {hit_case['src_text'][:80]}...")
    print(f"      B: {hit_case['dst_text'][:80]}...")
    print(f"      {hit_case['detail']}")
    pass_case = next(p for p in pairs if not p["hit"] and p["exp"] == "entail")
    print(f"  [正确放行] {pass_case['src']} -> {pass_case['dst']}（数值相同，互相支持）")
    print(f"      A: {pass_case['src_text'][:80]}...")
    print(f"      B: {pass_case['dst_text'][:80]}...")
    # 边界 case：投毒 claim 原文含多个原始数字（season 4 序号 / firehouse 51 裸数字
    # 被排除，仅 "24 episodes" 一个候选参与配对）
    bcase = next((p for p in should_intercept
                  if len(_extract_numbers(p["src_text"])) == 1
                  and len(re.findall(r"\d", p["src_text"])) >= 2), None)
    if bcase:
        cands = [f"{x['value']:g}{x['unit'] or ''}({x['kind']})"
                 for x in _extract_numbers(bcase["src_text"])]
        print("  [边界 case] 候选过滤（原文含 >=2 个数字，但仅 1 个候选参与）")
        print(f"      A: {bcase['src_text'][:80]}...")
        print(f"      -> 候选: {cands}")
        print("      （'season 4' 序号排除，'firehouse 51' 裸数字排除）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq", choices=["nq", "hotpotqa", "msmarco"])
    ap.add_argument("--num_queries", type=int, default=3)
    ap.add_argument("--synthetic-normal", type=int, default=2)
    args = ap.parse_args()
    main(args.dataset, args.num_queries, args.synthetic_normal)
