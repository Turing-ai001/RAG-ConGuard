"""关系抽取模块（P1）：NLI 主干 + LLM 低置信兜底（方案 C）。"""
from relation.extractor import RelationExtractor
from relation.llm_judge import LLMJudge
from relation.nli import NLIScorer
from relation.types import Relation, RelationType

__all__ = ["RelationExtractor", "LLMJudge", "NLIScorer", "Relation", "RelationType"]
