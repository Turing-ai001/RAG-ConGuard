# -*- coding: utf-8 -*-
"""E8 关键对照实验（本地重放）：逐文档 LOO 影响 vs 联盟级删除影响。

数据来源：output/eval_pipeline/eval_p2.json 的 rows（含 claims/edges/sims/coalitions，
由服务器端评估收集）。本脚本用相同 trust-传播/答案支持公式重放：
    single-doc LOO：删除该文档对应的单个 claim 节点 → max_a 相对影响
    coalition removal：删除整个联盟 → 影响
输出表：投毒文档组内 avg LOO vs avg coalition impact（按 1 / 3+ 篇组规模分组）。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coalition.trust import answer_support, trust_weights   # noqa: E402
from graph import ClaimGraph                                 # noqa: E402
from relation.types import Relation, RelationType            # noqa: E402


def _relist(edges):
    """JSON 序列化的边（dict）→ Relation 列表。"""
    out = []
    for e in edges:
        out.append(Relation(e["src"], e["dst"], RelationType(e["type"]),
                            float(e.get("score", 0.5)), e.get("provenance", "nli")))
    return out


def impact_after_removal(claims, edges, sims, answer_ids, removed):
    """删除 removed（claim_id 集合）后的相对影响（与 eval_p2 同式）。"""
    keep = [c for c in claims if c["claim_id"] not in removed]
    keep_edges = [e for e in edges if e["src"] not in removed and e["dst"] not in removed]
    if not keep:
        return 0.0
    g2 = ClaimGraph(keep, _relist(keep_edges))
    w = trust_weights(g2)
    idx = {c["claim_id"]: i for i, c in enumerate(claims)}
    sub = sims[[idx[c["claim_id"]] for c in keep]]
    s_full = None  # 用删除前完整图分数作为分母（相对影响口径）
    return _rel_impact(claims, edges, sims, answer_ids, keep=keep, w=w, sub=sub)


def _s_vals(claims, edges, sims, answer_ids):
    g = ClaimGraph(claims, _relist(edges))
    w = trust_weights(g)
    return answer_support(claims, w, sims, answer_ids)


def _rel_impact(claims, edges, sims, answer_ids, keep=None, w=None, sub=None):
    s2 = answer_support(keep, w, sub, answer_ids)
    s0 = _s_vals(claims, edges, sims, answer_ids)
    denom = max(s0.values()) or 1.0
    return max((s0.get(a, 0.0) - s2.get(a, 0.0)) for a in answer_ids) / denom


def main(src: str):
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    rows = data["rows"]
    agg = defaultdict(lambda: {"loo": [], "coal": []})   # 组大小 -> 列表
    n_q = 0
    for r in rows:
        claims, edges, sims, coalitions = r["claims"], r["edges"], r["sims"], r["coalitions"]
        if not coalitions:
            continue
        n_q += 1
        aid = r["answer_ids"]
        sims = np.asarray(sims, dtype=np.float64)
        # 逐文档 LOO（仅投毒联盟内的文档）
        for c in coalitions:
            size = len(c["claims"])
            if c["purity"] < 0.8:          # 只看投毒组
                continue
            for cid in c["claims"]:
                imp_doc = impact_after_removal(claims, edges, sims, aid, {cid})
                agg[size]["loo"].append(imp_doc)
            imp_coal = impact_after_removal(claims, edges, sims, aid, set(c["claims"]))
            agg[size]["coal"].append(imp_coal)

    print(f"queries analyzed: {n_q}\n")
    print(f"{'group size':<10}{'n_groups':>9}{'avg LOO (doc)':>16}{'avg coalition':>15}{'ratio':>8}")
    for size in sorted(agg):
        loo, coal = np.mean(agg[size]["loo"]), np.mean(agg[size]["coal"])
        n = len(agg[size]["coal"])
        print(f"{size:<10}{n:>9}{loo:>16.4f}{coal:>15.4f}{coal / max(loo, 1e-9):>8.2f}")
    # 汇总（所有投毒组）
    all_loo = [v for k in agg for v in agg[k]["loo"]]
    all_coal = [v for k in agg for v in agg[k]["coal"]]
    print(f"\noverall: LOO mean={np.mean(all_loo):.4f}  coalition mean={np.mean(all_coal):.4f}  ratio={np.mean(all_coal) / max(np.mean(all_loo), 1e-9):.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "output/eval_pipeline/eval_p2.json"))
