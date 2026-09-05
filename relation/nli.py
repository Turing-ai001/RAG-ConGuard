"""NLI 主干打分（方案 A）：cross-encoder 三分类（entailment/neutral/contradiction）。

模型：默认 cross-encoder/nli-deberta-v3-base（MNLI 微调 cross-encoder，
旧项目 RAG-ConGuard 已验证同款）。对 (premise, hypothesis) 打分，
返回三分类 softmax 概率 + 判定类型。

复用旧项目 NLIScorer 骨架（batch 循环、model.config.id2label 动态读取），
扩展点：返回完整三分类概率（旧项目只取 contradiction 类概率）。
"""
from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _AVAILABLE = True
except Exception:  # pragma: no cover
    _AVAILABLE = False

from config import DEVICE, NUMERIC_CONFLICT_HEURISTIC
from relation.types import NLI_LABELS, NLI_TO_RELATION, Relation, RelationType

# 可用 HF_NLI_MODEL 环境变量覆盖（本地缓存路径或替代模型；服务器上指向 models_cache）
DEFAULT_NLI_MODEL = os.environ.get("HF_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")


@dataclass
class NLIScorer:
    """cross-encoder NLI 打分器。score_pairs 返回 (probs, types, relations)。"""

    model_name: str = DEFAULT_NLI_MODEL
    device: str = DEVICE
    batch_size: int = 32
    max_length: int = 512
    local_files_only: bool = None   # None=跟随 HF_OFFLINE 环境变量

    def __post_init__(self) -> None:
        if not _AVAILABLE:
            raise RuntimeError("transformers/torch not installed; NLI unavailable")
        offline = os.environ.get("HF_OFFLINE", "").lower() in ("1", "true", "yes")
        if self.local_files_only is None:
            self.local_files_only = offline
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, local_files_only=self.local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, local_files_only=self.local_files_only)
        self.model.to(self.device)
        self.model.eval()
        # 标签顺序从模型配置读取，不硬编码（兼容 int/str 两种键）
        id2label = getattr(self.model.config, "id2label", None)
        if id2label:
            self.labels = []
            for i in range(len(id2label)):
                val = id2label.get(i, id2label.get(str(i), f"label_{i}"))
                self.labels.append(str(val).lower())
        else:
            self.labels = list(NLI_LABELS)
        self.label_index = {lab: i for i, lab in enumerate(self.labels)}
        print(f"[NLIScorer] {self.model_name} on {self.device}, labels={self.labels}")

    # ---------- 打分 ----------
    @torch.no_grad()
    def score_pairs(self, premises: Sequence[str],
                    hypotheses: Sequence[str]) -> np.ndarray:
        """(premise, hypothesis) → 三分类 softmax 概率矩阵 (n, 3)。"""
        if len(premises) != len(hypotheses):
            raise ValueError("premises/hypotheses length mismatch")
        if not premises:
            return np.zeros((0, len(self.labels)), dtype=np.float64)

        probs = []
        for start in range(0, len(premises), self.batch_size):
            end = min(start + self.batch_size, len(premises))
            bp, bh = premises[start:end], hypotheses[start:end]
            inputs = self.tokenizer(
                list(bp), list(bh), padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits
            p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            probs.append(p)
        return np.concatenate(probs, axis=0)

    # ---------- 关系化 ----------
    def classify(self, premises: Sequence[str], hypotheses: Sequence[str],
                 threshold: float = 0.7) -> list[tuple[RelationType, float, np.ndarray]]:
        """三分类 → 关系类型。返回 [(type, confidence, probs)]。

        高置信（max prob >= threshold）→ 类型由 NLI 决定；
        低置信 → 类型 UNRELATED（由上层 LLM 兜底覆盖）。

        数值归一化启发式（步骤2）：打分前对每对做 numeric_conflict_check，
        命中对跳过 NLI 模型，直接返回 (CONTRADICT, 1.0, one-hot probs)。
        """
        n = len(premises)
        if n == 0:
            return []
        hit_idx, keep_idx = [], []
        if NUMERIC_CONFLICT_HEURISTIC:
            for i, (p, h) in enumerate(zip(premises, hypotheses)):
                if numeric_conflict_check(p, h) is not None:
                    hit_idx.append(i)
                else:
                    keep_idx.append(i)
        probs = np.zeros((n, len(self.labels)), dtype=np.float64)
        if keep_idx:
            sub = self.score_pairs([premises[i] for i in keep_idx],
                                   [hypotheses[i] for i in keep_idx])
            for k, i in enumerate(keep_idx):
                probs[i] = sub[k]
        ci = self.label_index.get("contradiction", 0)
        for i in hit_idx:
            probs[i, ci] = 1.0    # 数值冲突 → contradiction 类 one-hot（conf=1.0）
        out = []
        for p in probs:
            lab = self.labels[int(p.argmax())]
            conf = float(p.max())
            t = NLI_TO_RELATION.get(lab, RelationType.UNRELATED)
            if conf < threshold:
                t = RelationType.UNRELATED
            out.append((t, conf, p))
        return out

    def edge_relations(self, src_texts: Sequence[str], dst_texts: Sequence[str],
                       src_ids: Sequence[str], dst_ids: Sequence[str],
                       threshold: float = 0.7) -> list[Relation]:
        """批量构建 Relation 边（src → dst），供图构建直接使用。"""
        classified = self.classify(src_texts, dst_texts, threshold)
        return [
            Relation(s, d, t, conf, provenance="nli")
            for (s, d), (t, conf, _p) in zip(zip(src_ids, dst_ids), classified)
        ]


# ==========================================================================
# 数字归一化启发式（P1 步骤2）：NLI 打分前拦截"同实体 + 不同数值"冲突对
# ==========================================================================
# 动机：MNLI 模型对数值冲突（"24集" vs "23集"、"2023年" vs "2024年"）存在
# 跨域盲区 —— 126 样本实测此类对 50% 被判为 unrelated（模型视为 neutral）。
# 纯正则 + 字符串操作实现，无新模型、无外部依赖；函数独立，可单测。
#
# 设计（每步都防误伤）：
#   1) 提取数值候选：正则 + 上下文窗口（前后 5 字符）+ 单位词表（英/中）
#   2) 排除噪声数字：序号（"season 4"）、裸数字（"firehouse 51"）
#   3) 冲突判定：同实体锚点（单位同词 / 上下文相似 / 共享内容词）+ 数值不同
#   4) 防误伤：数值差 <1% 视为 low_severity 放行（年份 4 位整数除外，
#      2023 vs 2024 差 0.05% 但属确定冲突）；单 claim 候选 >= 3 个不触发
#      （避免范围描述如 "1000、2000、3000 人" 误判）

# 单位词表：key = 原文单位（匹配用），value = 归一化单位族（跨词形对齐用）。
# 英文需先小写后匹配；匹配时按长度降序遍历（"episodes" 先于 "episode"、
# "首歌" 先于 "首"、"美元" 先于 "元"）。
_UNIT_FAMILIES = [
    ("episodes", "episode"), ("episode", "episode"), ("eps", "episode"),
    ("seasons", "season"), ("season", "season"),
    ("years", "year"), ("year", "year"), ("yr", "year"),
    ("months", "month"), ("month", "month"),
    ("days", "day"), ("day", "day"),
    ("hours", "hour"), ("hour", "hour"),
    ("minutes", "minute"), ("minute", "minute"),
    ("people", "people"), ("person", "people"), ("persons", "people"),
    ("percent", "percent"), ("%", "percent"),
    ("members", "member"), ("member", "member"),
    ("students", "student"), ("student", "student"),
    ("points", "point"), ("point", "point"),
    ("dollars", "dollar"), ("dollar", "dollar"),
    ("million", "million"), ("billion", "billion"), ("thousand", "thousand"),
    # 中文单位（均在数字后，如 "24集"、"3.5亿"、"增长了15%"）
    ("集", "episode"), ("年", "year"), ("月", "month"), ("日", "day"), ("天", "day"),
    ("岁", "age"),          # 年龄独立成族，避免与年份（year）误配
    ("人", "people"), ("万", "wan"), ("亿", "yi"), ("元", "yuan"),
    ("美元", "dollar"), ("公里", "km"), ("千米", "km"), ("米", "m"),
    ("首歌", "song"), ("首", "song"), ("部", "film"), ("本", "book"),
]

# 运行时按长度降序排序，保证最长单位优先匹配（"episodes" > "episode"）
_UNIT_FAMILIES_SORTED = sorted(_UNIT_FAMILIES,
                               key=lambda kv: len(kv[0]), reverse=True)

# 序号词：数字前紧邻这些词 → 序号（"season 4" 的 4 不是事实性数值，排除）
_ORDINAL_WORDS = {"season", "series", "part", "chapter", "round", "volume",
                  "vol", "book", "phase", "level", "episode", "stage"}

# 停用词：提取"共享内容词"时过滤（how/many 等 query 疑问词、功能词）
_STOPWORDS = set("""
a an the of to in for on at with by from and or not but this that these those
it its as have has had can could would should will may might than then there
their they we you he she them his her also very just about into over under
after before between during out up down off all any both each few more most
other some such no nor only own same so too how many what which when where
who why do does did is are was were be been being s t re ve ll d m
""".split())

# 数值正则：支持千分位 "1,000" 与小数 "3.2"；负向断言避免匹配单词内数字。
# 注意：后向断言保留 "."（防 "3.5" 的 "5" 被单独匹配），前向断言不含 "."
# （"is 23." 句号结尾的数字必须能匹配）。
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![A-Za-z0-9])")


def _match_unit(after: str, pre: str) -> Optional[str]:
    """匹配数字后紧邻的单位词（after 为数字后的原文长切片），返回单位族。

    例：after=" episodes, engaging" → "episode"；after="集，共24" → "episode"；
        after="%" → "percent"；after="." → None；pre 后缀 "$24" → "dollar"。
    注意：单位词可能超过 5 字符窗口（"episodes" 8 字符），因此 after 取
    数字后 24 字符切片，上下文锚点窗口（5 字符）与单位提取分离。
    """
    tail = after.lstrip()
    for unit, family in _UNIT_FAMILIES_SORTED:
        if len(unit) > 1:  # 英文/多字词：需要词边界，避免 "episodesx" 误配
            if not tail.lower().startswith(unit):
                continue
            rest = tail[len(unit):]
            if rest[:1].isalpha():
                continue  # 单位后紧跟字母 → 不是独立单位词（"24episodesx"）
        else:  # 单字符（"%"、中文单字）：直接前缀匹配
            if not tail.startswith(unit):
                continue
        return family
    # pre 后缀兜底：货币前置 "$24"
    if pre.rstrip().endswith("$"):
        return "dollar"
    return None


def _extract_numbers(text: str) -> list[dict]:
    """提取数值候选。返回 [{value, pre, post, unit, kind}]。

    kind ∈ {"quantity"（有单位，如 "24 episodes"）, "answer"（be 动词后，
    如 "is 23"）, "year"（4 位年份，如 "released in 2023"）}；
    序号（"season 4"）与裸数字（"firehouse 51"）直接排除。
    """
    out = []
    for m in _NUMBER_RE.finditer(text):
        start, end = m.span()
        value = float(m.group(0).replace(",", ""))
        pre = text[max(0, start - 5):start]      # 上下文窗口：前 5 字符（实体锚点）
        post = text[end:end + 5]                 # 上下文窗口：后 5 字符（实体锚点）
        after = text[end:end + 24]               # 单位提取：数字后长切片

        # 防误伤①：序号排除 —— "season 4"、"series 2" 中的数字不是事实性数值。
        # 取数字前 20 字符检查完整前词（5 字符窗口会把 "season" 截成 "ason"）。
        pre20 = text[max(0, start - 20):start]
        m_pre = re.search(r"([A-Za-z]+)\s*$", pre20)
        if m_pre and m_pre.group(1).lower() in _ORDINAL_WORDS:
            continue

        unit = _match_unit(after, pre)
        if unit:
            kind = "quantity"
        # 年份型：4 位整数 + "in/of" 前置（"released in 2023"）；"2023年" 已被
        # 单位匹配归入 quantity(year)，此处覆盖无单位年份。年份独立成类，
        # 只与年份配对（"24岁" vs "2024年" 不误配）。
        elif (value.is_integer() and 1000 <= value <= 9999
              and re.search(r"\b(?:in|of)\s*$", pre)):
            unit, kind = "year", "year"
        elif re.search(r"\b(?:is|are|was|were|equals?)\s*$", pre):
            kind = "answer"    # 答案型数字："the answer is 23" / "contains is 24"
        else:
            continue           # 裸数字（"firehouse 51"）：无单位无 be 动词 → 排除
        out.append({"value": value, "pre": pre, "post": post,
                    "unit": unit, "kind": kind})
    return out


def _shared_content_words(a: str, b: str) -> set[str]:
    """两个 claim 的共享内容词（>=3 字母，去停用词）—— 轻量"同实体"信号。"""
    wa = {w for w in re.findall(r"[A-Za-z]{3,}", a.lower()) if w not in _STOPWORDS}
    wb = {w for w in re.findall(r"[A-Za-z]{3,}", b.lower()) if w not in _STOPWORDS}
    return wa & wb


def _severity_ok(va: float, vb: float, ua: Optional[str], ub: Optional[str]) -> bool:
    """数值差异显著性。相对差 <1% → low_severity 放行（走 NLI）；
    例外：双方 4 位年份（2023 vs 2024 相对差 0.05% 但属确定冲突）。
    """
    if ua == "year" and ub == "year" and 1000 <= va <= 9999 and 1000 <= vb <= 9999:
        return True
    rel = abs(va - vb) / max(abs(va), abs(vb), 1e-9)
    return rel >= 0.01


def _anchor_match(ta: str, pa: str, ua: Optional[str],
                  tb: str, pb: str, ub: Optional[str], shared: set[str]) -> bool:
    """实体锚点相似判定：
        quantity vs quantity : 单位族相同 → 同实体（"24 episodes" vs "23 episodes"）
        answer   vs answer   : 数值前上下文相似 >0.8（"is 23" vs "is 24"），
                               或共享内容词 >=1
        quantity vs answer   : 共享内容词 >=1（"is 23" vs "total of 24 episodes"
                               靠 query 实体词 chicago/fire/episodes 对齐）
        year     vs year     : 年份与年份（"2023" vs "2024"）；年份不与任何
                               其他类型配对（防 "24岁" vs "2024年" 误配）
    """
    if ta == "year" or tb == "year":
        return ta == "year" and tb == "year"
    if ta == "quantity" and tb == "quantity":
        return ua == ub
    if ta == "answer" and tb == "answer":
        if difflib.SequenceMatcher(None, pa, pb).ratio() > 0.8:
            return True
        return bool(shared)
    return bool(shared)   # quantity vs answer


def numeric_conflict_check(claim_a: str, claim_b: str) -> Optional[tuple[bool, str]]:
    """数字归一化启发式：检测"同实体 + 不同数值"冲突对。

    返回：
        (True, detail_str) —— 检测到数值冲突。调用方跳过 NLI，
                             直接生成 contradiction 边，置信度 1.0。
        None               —— 未检测到冲突（或属防误伤场景），走正常 NLI 流程。

    独立函数：只依赖两个 claim 文本，不依赖图构建/检索上下文，可单测。
    """
    ca = _extract_numbers(claim_a)
    cb = _extract_numbers(claim_b)
    # 防误伤②：任一 claim 候选数值 >= 3 个 → 范围描述（"1000、2000、3000 人"），不触发
    if len(ca) >= 3 or len(cb) >= 3:
        return None
    if not ca or not cb:
        return None

    shared = _shared_content_words(claim_a, claim_b)
    for x in ca:
        for y in cb:
            if x["value"] == y["value"]:
                continue                      # 数值相同 → 不是冲突（"24集" vs "24集"）
            if not _severity_ok(x["value"], y["value"], x["unit"], y["unit"]):
                continue                      # 防误伤③：low_severity（1000 vs 1005）放行
            if _anchor_match(x["kind"], x["pre"], x["unit"],
                             y["kind"], y["pre"], y["unit"], shared):
                detail = (f"numeric conflict: {x['value']:g}{x['unit'] or ''} "
                          f"(anchor '{x['pre'].strip()}') vs "
                          f"{y['value']:g}{y['unit'] or ''} "
                          f"(anchor '{y['pre'].strip()}')")
                return (True, detail)
    return None
