"""防御决策层（P3）：联盟风险学习器 + 风险约束上下文选择 + 安全生成。

从 P2 联盟特征（内聚/佐证/反事实影响/控制力）到防御动作：
    风险学习器 → 风险分 → 高风险联盟对应文档剔除 → （全剔则拒答）→ 安全生成。
"""
from defense.risk import LearnedRisk, RuleRisk
from defense.pipeline3 import DefensePipeline

__all__ = ["RuleRisk", "LearnedRisk", "DefensePipeline"]
