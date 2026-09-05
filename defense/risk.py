"""联盟风险学习器（P3）。

输入：P2 联盟特征（size / cohesion / external_corroboration / external_refute /
impact / control），分为两种：
    RuleRisk    零学习规则基准：risk = control（影响 × 佐证缺失），阈值即可防御
    LearnedRisk 学习型（sklearn MLP，3 层小网络）：监督信号 = 联盟纯度 >= 0.8
                （弱标签，来自投毒数据标注），按 query 划分训练/验证避免泄漏

输出：risk ∈ [0, 1]（越大越可疑），供 filter 剔除/拒答。
"""
from __future__ import annotations

import numpy as np

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except Exception:   # 无 sklearn 时退化规则版
    _SKLEARN = False

_FEATURES = ["size", "cohesion", "external_corroboration",
             "external_refute", "impact", "control"]

_WEIGHTS = {          # RuleRisk 加权组合（无需训练；论文可解释）
    "size": 1.0, "cohesion": 1.5, "external_corroboration": -2.0,
    "external_refute": 0.5, "impact": 2.0, "control": 3.0,
}


def feature_matrix(coalitions: list[dict]) -> np.ndarray:
    """联盟记录列表 → 特征矩阵（列序 = _FEATURES）。"""
    return np.array([[float(c.get(f, 0.0)) for f in _FEATURES]
                     for c in coalitions], dtype=float)


class RuleRisk:
    """规则风险分：weighted sum（正则化到 [0,1] 用 sigmoid 压缩）。"""

    def __init__(self):
        self.kind = "rule"

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        raw = np.array([sum(_WEIGHTS[f] * feat[i]
                            for i, f in enumerate(_FEATURES))
                        for feat in feats])
        return 1.0 / (1.0 + np.exp(-raw))          # sigmoid → [0,1]


class LearnedRisk:
    """sklearn MLP 风险学习器。fit(pool, labels) → predict_proba(feats)。"""

    def __init__(self, random_state: int = 42):
        if not _SKLEARN:
            raise RuntimeError("scikit-learn 未安装；请用 RuleRisk")
        self.kind = "mlp"
        self._scaler = StandardScaler()
        self._model = MLPClassifier(
            hidden_layer_sizes=(16, 8), max_iter=500,
            random_state=random_state,
        )

    def fit(self, pool: list[dict], labels: list[int]) -> "LearnedRisk":
        feats = feature_matrix(pool)
        self._scaler.fit(feats)
        self._model.fit(self._scaler.transform(feats), labels)
        return self

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._scaler.transform(feats))[:, 1]

    # ---------- 持久化（P4 攻防循环复用） ----------
    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"scaler": self._scaler, "model": self._model}, f)

    @classmethod
    def load(cls, path: str) -> "LearnedRisk":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        risk = cls()
        risk._scaler, risk._model = obj["scaler"], obj["model"]
        return risk


def make_risk(kind: str = "mlp") -> RuleRisk | LearnedRisk:
    """工厂：kind ∈ rule | mlp（无 sklearn 自动退化 rule）。"""
    if kind == "mlp" and _SKLEARN:
        return LearnedRisk()
    return RuleRisk()
