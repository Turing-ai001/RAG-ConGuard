"""联盟分析模块（P2）：联盟特征（内聚/佐证）+ 反事实影响（信任传播/答案支持）。

支持联盟检测核心①：有符号图（graph.build）→ 联盟发现 → 结构特征（analyze）
→ 图聚合反事实影响（trust）。
"""
from coalition.analyze import coalition_features
from coalition.trust import answer_support, counterfactual_impacts, trust_weights

__all__ = ["coalition_features", "trust_weights", "answer_support",
           "counterfactual_impacts"]
