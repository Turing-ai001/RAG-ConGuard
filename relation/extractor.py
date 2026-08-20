"""关系抽取器（方案 C：NLI 主干 + LLM 低置信兜底）。

流程：
    1. NLI 主干（DeBERTa-NLI）对全部 claim 对打分（方案 A）
    2. 高置信（max prob >= NLI_CONFIDENCE_THRESHOLD）→ 直接采用 NLI 结果
    3. 低置信 pair → LLM-as-judge 重判（方案 B 兜底）
    4. 融合输出 Relation 列表，标注 provenance（nli / llm）

论文叙事写方案 C（融合），实现上核心权重在 NLI —— 低置信 pair 占比
通常 < 10%，LLM 只处理该子集，成本可控。
"""
from __future__ import annotations

from typing import Optional

from config import NLI_CONFIDENCE_THRESHOLD, NLI_LLM_FUSION_ALPHA
from relation.llm_judge import LLMJudge
from relation.nli import NLIScorer
from relation.types import Relation, RelationType


class RelationExtractor:
    def __init__(self, nli: NLIScorer, llm_judge: Optional[LLMJudge] = None,
                 threshold: float = NLI_CONFIDENCE_THRESHOLD):
        self.nli = nli
        self.judge = llm_judge      # None 时纯 NLI（消融/无 LLM 环境可用）
        self.threshold = threshold

    # ---------- 主接口 ----------
    def extract_all(self, claims: list[dict],
                    low_conf_batch: int = 16) -> list[Relation]:
        """claims: [{claim_id, text}] → 全部有序 pair 的关系列表（不含自环）。"""
        n = len(claims)
        if n < 2:
            return []
        ids = [c["claim_id"] for c in claims]
        texts = [c["text"] for c in claims]
        src_ids, dst_ids = [], []
        src_texts, dst_texts = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src_ids.append(ids[i]); dst_ids.append(ids[j])
                    src_texts.append(texts[i]); dst_texts.append(texts[j])

        # 1) NLI 主干（classify 内部已做数值启发式预检，命中对直接 contradict）
        classified = self.nli.classify(src_texts, dst_texts, self.threshold)

        edges: list[Relation] = []
        llm_src, llm_dst, llm_si, llm_di = [], [], [], []
        nli_scores: dict[tuple[str, str], float] = {}   # 低置信对的 NLI 分数（融合用）
        for (si, di, st, dt), (t, conf, _p) in zip(
                zip(src_ids, dst_ids, src_texts, dst_texts), classified):
            if t is not RelationType.UNRELATED:
                edges.append(Relation(si, di, t, conf, provenance="nli"))
            elif conf < self.threshold and self.judge is not None:
                # 低置信 → LLM 兜底（记下 NLI 分数供融合）
                llm_si.append(si); llm_di.append(di)
                llm_src.append(st); llm_dst.append(dt)
                nli_scores[(si, di)] = conf
            else:
                # 高置信 neutral → 无关联（NLI 判为确实无关）
                edges.append(Relation(si, di, RelationType.UNRELATED, conf, provenance="nli"))

        # 2) LLM 兜底（低置信子集）：融合分数 alpha*s_NLI + (1-alpha)*s_LLM
        #    极性/类型以 LLM 判定为准，分数为 NLI 与 LLM 置信度的加权和。
        if llm_si:
            judged = self.judge.judge_edges(llm_src, llm_dst, llm_si, llm_di,
                                            batch_size=low_conf_batch)
            for e in judged:
                s_nli = nli_scores.get((e.src, e.dst), 0.0)
                e.score = NLI_LLM_FUSION_ALPHA * s_nli \
                    + (1 - NLI_LLM_FUSION_ALPHA) * e.score
                edges.append(e)
        return edges

    def stats(self, edges: list[Relation]) -> dict:
        n_llm = sum(1 for e in edges if e.provenance == "llm")
        by_type = {}
        for e in edges:
            by_type[e.type.value] = by_type.get(e.type.value, 0) + 1
        return {
            "n_edges": len(edges),
            "llm_fallback_ratio": round(n_llm / len(edges), 4) if edges else 0.0,
            "by_type": by_type,
        }
