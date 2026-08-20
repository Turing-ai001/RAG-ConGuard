"""关系类型定义：claim 之间边的类型、极性、来源。

五种边类型（论文口径）：
    SUPPORT     支持(+，宽泛正面，LLM judge 可用)
    REFUTE      反驳(−，宽泛负面，LLM judge 可用)
    ENTAIL      蕴含(+，严格逻辑蕴含，NLI entailment)
    CONTRADICT  矛盾(−，事实冲突，NLI contradiction)
    UNRELATED   无关联(0)

极性（图上的符号）：
    + : SUPPORT / ENTAIL
    - : REFUTE / CONTRADICT
    0 : UNRELATED

工程映射（方案 C：NLI 主干 + LLM 兜底）：
    - NLI 高置信：entailment → ENTAIL(+)，contradiction → CONTRADICT(−)，neutral → UNRELATED
    - NLI 低置信（LLM 兜底）：模型输出 SUPPORT / REFUTE / ENTAIL / CONTRADICT / UNRELATED
"""
from __future__ import annotations

from enum import Enum

from config import NLI_CONFIDENCE_THRESHOLD  # noqa: F401  (re-export 供模块间统一)


class RelationType(str, Enum):
    SUPPORT = "support"        # (+)
    REFUTE = "refute"          # (-)
    ENTAIL = "entail"          # (+)
    CONTRADICT = "contradict"  # (-)
    UNRELATED = "unrelated"    # (0)

    @property
    def polarity(self) -> int:
        """边符号：+1 支持 / -1 反驳 / 0 无关联。"""
        return {
            RelationType.SUPPORT: 1,
            RelationType.ENTAIL: 1,
            RelationType.REFUTE: -1,
            RelationType.CONTRADICT: -1,
            RelationType.UNRELATED: 0,
        }[self]

    @property
    def label(self) -> str:
        return {
            RelationType.SUPPORT: "支持(+)",
            RelationType.REFUTE: "反驳(−)",
            RelationType.ENTAIL: "蕴含(+)",
            RelationType.CONTRADICT: "矛盾(−)",
            RelationType.UNRELATED: "无关联",
        }[self]


# NLI 模型标签顺序（cross-encoder 系列：0=contradiction, 1=entailment, 2=neutral）
# 不硬编码：从 model.config.id2label 读取；此表仅作映射参考。
NLI_LABELS = ["contradiction", "entailment", "neutral"]

NLI_TO_RELATION = {
    "entailment": RelationType.ENTAIL,
    "contradiction": RelationType.CONTRADICT,
    "neutral": RelationType.UNRELATED,
}

class Relation:
    """一条 claim 边。src → dst 的关系。"""

    __slots__ = ("src", "dst", "type", "score", "provenance")

    def __init__(self, src: str, dst: str, type_: RelationType,
                 score: float, provenance: str = "nli"):
        self.src = src
        self.dst = dst
        self.type = type_
        self.score = score          # 置信度 [0,1]
        self.provenance = provenance  # "nli" | "llm"

    @property
    def polarity(self) -> int:
        return self.type.polarity

    def to_dict(self) -> dict:
        return {
            "src": self.src, "dst": self.dst,
            "type": self.type.value, "polarity": self.polarity,
            "score": round(self.score, 3), "provenance": self.provenance,
        }

    def __repr__(self) -> str:
        return (f"Relation({self.src} --[{self.type.label}|{self.score:.2f}|"
                f"{self.provenance}]--> {self.dst})")
