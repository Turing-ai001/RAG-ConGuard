"""数字归一化启发式单元测试（P1 步骤2）。

函数独立于图构建上下文，直接测 relation.nli.numeric_conflict_check。
兼容 pytest（def test_*）与直接运行（python tests/test_numeric_conflict.py）。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from relation.nli import numeric_conflict_check  # noqa: E402


# ---------- 正确拦截 ----------
def test_en_quantity_conflict():
    """英文量词型：同单位 episodes + 不同数值 → 拦截。"""
    r = numeric_conflict_check(
        "The series has a total of 24 episodes.",
        "The series has a total of 23 episodes.")
    assert r is not None and r[0] is True
    assert "episode" in r[1]


def test_cn_quantity_conflict():
    """中文量词型："该剧共24集" vs "该剧共23集"（用户示例）。"""
    r = numeric_conflict_check("该剧共24集。", "该剧共23集。")
    assert r is not None and r[0] is True


def test_year_conflict():
    """年份差 1（2023 vs 2024，相对差 0.05% <1%）仍须拦截。"""
    r = numeric_conflict_check(
        "The film was released in 2023.",
        "The film was released in 2024.")
    assert r is not None and r[0] is True


def test_percent_conflict():
    """百分比：'增长了15%' vs '下降3.2%' → 同单位 percent → 拦截。"""
    r = numeric_conflict_check("Sales increased by 15% last year.",
                               "Sales increased by 3.2% last year.")
    assert r is not None and r[0] is True


def test_answer_type_conflict():
    """答案型（be 动词前）：'is 23' vs 'is 24' → 上下文相似 → 拦截。"""
    r = numeric_conflict_check(
        "the answer to the query is 23, confirmed by sources.",
        "the answer to the query is 24, confirmed by sources.")
    assert r is not None and r[0] is True


def test_quantity_vs_answer_conflict():
    """量词 vs 答案混合："total of 24 episodes" vs "is 23" → 共享词 → 拦截。"""
    r = numeric_conflict_check(
        "The fourth season of Chicago Fire contains a total of 24 episodes.",
        "the answer to \"how many episodes are in chicago fire season 4\" is 23")
    assert r is not None and r[0] is True


# ---------- 正确放行 ----------
def test_user_counterexample_different_entity():
    """用户反例："该剧共24集" vs "播出时间为2023年" → 单位族不同 → 放行。"""
    assert numeric_conflict_check("该剧共24集。", "播出时间为2023年。") is None


def test_en_different_entity():
    """不同实体英文：episodes vs 建成年份 → 单位族不同 → 放行。"""
    assert numeric_conflict_check("The film has 24 episodes.",
                                  "The building was built in 2005.") is None


def test_same_value_pass():
    """数值相同（24 vs 24，同组互相支持）→ 放行。"""
    assert numeric_conflict_check("It contains 24 episodes.",
                                  "It also contains 24 episodes.") is None


def test_low_severity_pass():
    """防误伤：1000 vs 1005 相对差 0.5% <1% → low_severity 放行。"""
    assert numeric_conflict_check("The factory has 1000 workers.",
                                  "The factory has 1005 workers.") is None


# ---------- 边界 case ----------
def test_ordinal_excluded():
    """序号排除："season 4" 的 4 不参与，24 vs 23 仍触发。"""
    r = numeric_conflict_check(
        "season 4 of chicago fire contains is 24.",
        "the answer to \"how many episodes are in chicago fire season 4\" is 23")
    assert r is not None and r[0] is True


def test_bare_number_excluded():
    """裸数字排除：firehouse 51 无单位无 be 动词 → 不参与配对 → 放行。"""
    assert numeric_conflict_check(
        "firehouse 51's team has 24 episodes of story.",
        "firehouse 52's team has 24 episodes of story.") is None


def test_many_numbers_pass():
    """范围描述防误伤：候选数值 >= 3 个 → 不触发。"""
    assert numeric_conflict_check(
        "It has approximately 1000, 2000 and 3000 members.",
        "It has approximately 1000, 2000 and 3000 members.") is None


def test_no_numbers_pass():
    """无数值 → 放行。"""
    assert numeric_conflict_check("No numbers here at all.",
                                  "Still no digits.") is None


if __name__ == "__main__":
    fns = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    passed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
