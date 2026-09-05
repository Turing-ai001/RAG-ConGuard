# PROJECT_AUDIT.md — RAG-ConGuard 服务器全面审计报告

> 审计日期：2026-08-25（Phase 0）
> 审计对象：AutoDL 服务器 `connect.nmb1.seetacloud.com:44793` 项目 `/root/autodl-tmp/RAG`
> 结论：代码=数据=模型=日志齐全，**可复现**；但实验体系存在 7 个硬伤（见 §12），
> 论文当前状态不可直接投稿。本地 `F:\论文\RAG` 与服务器核心 23 个文件 **md5 逐字节一致**（本地为源）。

---

## 1. 真实目录结构（服务器）

```
/root/autodl-tmp/RAG/
├── config.py                 # 全局配置（路径/模型/超参/P2_P3 常量/RAG_PROMPT）
├── pipeline.py               # 旧 P0 流水线（声明验证方向，非论文主链）
├── README.md                 # P0 里程碑（三层指标）说明
├── retrieval/                # 多视角检索
│   ├── embedder.py           #   BGE-base-en-v1.5（HF_EMBEDDING_MODEL 可覆盖）
│   ├── vector_store.py       #   FAISS IndexFlatIP + L2 归一 = cosine
│   ├── keyword.py            #   BM25（多进程分词 36s/600k）
│   ├── kg.py                 #   KG 视角（EntityKGStore；default_kg_store=空 → 报告配置关闭）
│   ├── fusion.py             #   RRF 融合（每视角 top-20 → top-5）
│   └── kb.py                 #   KnowledgeBase（检索入口+持久化+投毒文档加载）
├── claim/extractor.py        # LLM 主张抽取（M3 文中宣称，见 §3/§12：实际主链未用）
├── relation/                 # 关系抽取（方案 C：NLI 主干 + LLM 低置信兜底）
│   ├── nli.py                #   cross-encoder/nli-deberta-v3-base + 数值冲突启发式（24集 vs 23集）
│   ├── llm_judge.py          #   LLM-as-judge 重判低置信对，α=0.3 融合
│   ├── extractor.py          #   RelationExtractor.extract_all（全有序对）
│   └── types.py              #   Relation/RelationType
├── graph/build.py            # ClaimGraph：signed/unsigned/similarity + signed_semantic(8-25新增)
├── coalition/
│   ├── analyze.py            # 6 特征：size/cohesion/外佐证/外反驳/impact/control(析出)
│   └── trust.py              # 信任传播 p_c + 答案支持 s_G(A) + counterfactual_impacts(代理 CCI)
├── defense/
│   ├── filter.py             # S0：risk≥τ 整联盟剔除；全剔=拒答
│   ├── risk.py               # RuleRisk(加权sigmoid) / LearnedRisk(sklearn MLP 16-8, StandardScaler)
│   └── pipeline3.py          # DefensePipeline：检索→claim→图→联盟→风险→安全子集→生成/拒答
├── llm/                      # 后端 mock | hf | vllm；Qwen2.5-7B-Instruct (T=0, 512 tok)
├── scripts/                  # 旧实验入口（eval_p2/p3/p4, baselines, ablation_signed 等）
├── tests/                    # smoke_test.py, test_numeric_conflict.py
├── data/
│   ├── raw/                  # {nq,hotpotqa,msmarco}_{corpus,queries}.jsonl（BEIR 官方格式）
│   ├── poisoned/             # ~100 q/dataset × 5 adv_texts = 500 篇投毒文档（PoisonedRAG 格式）
│   └── index/{nq,hotpotqa,msmarco}_kb/
│       ├── docs.jsonl        #   600,500 行 = 600k 语料 + 500 投毒（doc_id 含 :poisoned:）
│       ├── index.faiss       #   BGE 向量（1.8GB）
│       ├── ids.json / bm25.pkl
├── output/eval_pipeline/     # 全部旧实验结果（10 个 JSON）
├── tests/
└── run_chain.sh              # 主链自动化（T2→T3→T4→T6→T5），run_supplement.sh（S1-S4）
```

环境：conda `rag`（py3.12, torch 2.5.1+cu124, transformers 5.15, faiss-cpu 1.15,
sentence-transformers 6.0, rank-bm25 0.2.2, accelerate 1.14, datasets 5.0.1, sklearn 有）。
GPU：RTX 4090 24.5GB（审计时 0% 空闲）。磁盘：/root/autodl-tmp 100G，已用 29G/余 72G。

## 2. 已实现模块

| 模块 | 状态 | 说明 |
|---|---|---|
| 多视角检索 | ✅ 完整 | BGE+FAISS + BM25 + RRF；KG 接口存在但报告配置禁用 |
| Claim 表示 | ⚠️ 偏离论文 | 主链 = 文档前 220 字符截断（claims_from_docs），**非** LLM 主张抽取；claim/extractor.py 存在但主链未调 |
| 有符号证据图 | ✅ 完整 | NLI 三分类阈值 0.7 + 数值冲突启发式 + LLM 兜底 α=0.3 融合；signed_semantic 双通道变体 8-25 新增（主链未启用） |
| 联盟发现 | ✅ | 正边连通分量（k=5 时精确可靠） |
| CCI（代理） | ⚠️ 代理实现 | trust-weights + 答案支持 + max_a 相对影响（**非** 生成 JSD 的反事实——论文 M6 形式化与实现不一致） |
| 风险学习器 | ✅ | MLP(16,8) + StandardScaler（fit 于训练池）；RuleRisk 规则基准 |
| 安全上下文选择 | ⚠️ 仅 S0 | 整联盟阈值剔除 → 全剔拒答（不满足论文"风险约束重建"表述） |
| 攻防循环 | ⚠️ 雏形 | eval_p4：A/B/C 三模板改写 + 逃脱样本回灌 + 重训 |
| LLM 生成 | ✅ | Qwen2.5-7B-Instruct bf16 T=0（hf 后端）；vllm 后端代码在（未装 vllm） |

## 3. 未实现模块（协议必须补）

- **Exact CCI**（真·生成分布 JSD，teacher-forced 对齐前缀）——完全缺失，Phase 3 补
- **安全上下文选择 S1/S2**（仅删最高风险联盟 / 贪婪重建）——Phase 2 补
- **正式 split（train/val/test 冻结文件）**——完全缺失（现按插入序 70/30）
- **Clean utility 实验**（无投毒查询集 + 良性误滤率）——完全缺失
- **强 baseline**：无 TrustRAG / ReliabilityRAG / RobustRAG / LOO 逐文档防御 / 旧版 RS+GS+CS
  ——服务器 grep 无任何相关代码（grep 'RS+GS\|TrustRAG\|ReliabilityRAG\|RobustRAG\|Joint-GCG' = 0 命中）
- **Unseen adaptive attack**（不参与训练的变体族）——缺失
- **跨模型（Llama-3.1-8B / Mistral/Gemma）**——models_cache 仅有 BGE/Qwen/NLI，需下载（15-20GB，磁盘余量 72G，安全）

## 4. 已存在实验（全部出于 output/eval_pipeline/ + /root/autodl-tmp/exp/）

| 实验 | 命令/日志 | 关键结果（真实） |
|---|---|---|
| T2 消融（90q/真实 judge） | ablation_signed.py / t2_ablation.log | 联盟纯度 signed 0.972 / unsigned 0.891 / similarity 0.850；冲突占比 0.124 vs 0.455 vs 0.000 |
| T3 检测+防御模拟 | eval_p2 / t3 | P=0.95 R=0.418 F1=0.425, AUC 0.623（纯 NLI 版） |
| S1 judge 版 eval_p2 | s1 日志 | 见 eval_p2.json（1.9MB，8-25 7:50） |
| T4 端到端 MLP 防御 | eval_p3 / t4 | AUC：learned 0.792 / rule(control) 0.194；无防御 ASR 0.857；τ∈{0.05..0.5} 全网格 ASR 0.143 / 正确率 0.057 / **拒答 1.000** |
| T6 基线+主链 | baselines.py，t6 | relevance 过滤 0% 改善；dup 过滤 0.76/0.93/0.70；无防御 ASR NQ 0.86 / HQ 0.99 / MS 0.82；检索命中 0.99/1.00/0.97 |
| T5 adaptive | eval_p4 / t5 | round0/1 逃脱 0.00 → round2 逃脱 1.00（τ=0.5，5q 小样本） |
| S2 细 τ 网格 | s2 | 全网格 refuse=1.0（确认防御过度而非阈值问题） |
| S3 更多轮攻防 | s3 | eval_p4.json（pool=89） |
| S4 signed_semantic | s4 | eval_p2.json 更新版 |

## 5. 模型与 checkpoint

- 模型缓存：`BAAI/bge-base-en-v1.5`（768d）、`Qwen/Qwen2.5-7B-Instruct`（4×safetensors，15G）、
  `cross-encoder/nli-deberta-v3-base`（5.6G）——全部验证通过（8-24 全测）
- **无 MLP checkpoint 文件**：T4 训练在每个进程内完成，未持久化（eval_p4 也无保存）
- **无 vllm**；无其他 LLM（Llama/Mistral/Gemma 缺失）

## 6. 数据集

| 数据集 | 语料 | 查询 | 投毒查询 | 投毒文档 |
|---|---|---|---|---|
| NQ | 600,000（test 子集语料） | 3,452（PoisonedRAG 的 test split） | 100 | 500（5×100） |
| HotpotQA | 600,000 | 5,000 | 100 | 500 |
| MS MARCO | 600,000 | 5,000 | 100 | 500 |

PoisonedRAG 输入格式：`{qid: {id, question, "correct answer", "incorrect answer", adv_texts[5]}}`
（这些查询/答案/投毒文本直接来自该基准）→ 已写入 KB（`data/index/{ds}_kb`，含 500 投毒）。
语料注入率 500/600,000 ≈ 0.08%。

## 7. 当前攻击

- 主攻击：PoisonedRAG 式 **Targeted Poisoning**（每查询 5 篇近重复投毒文本，注入语料，无防护时
  top-5 平均命中 3.88（NQ）/4.89（HQ）/3.12（MS）——攻击密度极高）
- 自适应变体（eval_p4）：A 互证版 / B 去协调版 / C 混合版；白盒最优风险信号（每轮对当前检测器打分）
- 无 Joint-GCG / 隐门 / 多阶段联合攻击代码（旧项目若曾实现，本服务器无残留）→ B 线攻击为公平起见应记录（缺），不能假装有

## 8. 当前 baseline

- B0 Vanilla（无防御）
- B1 relevance-filter（BGE 相似度阈值，τ=0.30 → 留存全部 → 0 改善）
- B2 dup-filter（剪枝相似度组，τ=0.85 → 0.76/0.93/0.70）
- （auc 另报告 rule/learned；没有任何发布防御、LOO 防御或旧版 RAG-ConGuard）

## 9. 当前指标

- 检索：retrieval ASR@5、多视角命中率（dense/keyword）
- 生成：generation ASR（clean_str + 双向包含 match）、正确率、拒答率
- 检测：联盟纯度 / 冲突占比 / AUC（手写 Mann-Whitney _auc）/ 规则 AUC
- 缺陷：无 clean utility、无良性误滤、无统计检验、无延迟/成本、ASR/refusal/accuracy 未做互斥主状态

## 10. 当前 train/val/test 划分

- **不存在正式 split 文件。**
- eval_p3：按 poisoned.json 字典插入序取前 70% query 训练、后 30% 评估（=210/90 共 300）
- baselines.py：100 查询/数据集（= 全量），**与 70 query 的防御评估不同池** —— 论文表 2/3 的不公平点
- 无 clean 查询集合
- 自适应变体生成以同一批 query → 变体可能跨入"评估"池 → 泄漏风险（pool 训练集与评估集重叠）
- 数值断言：文件数/数据总量已核实（300 投毒 query / 1500 adv 文本 / 3×600,500 文档）

## 11. 可复现性结论

- 全链自动化（run_chain.sh / run_supplement.sh），日志 10+ 份，中间产物全部落盘 → **可复现** ✅
- 但：①最终防御数据来自 eval_p3（70q），baseline 来自 t6（100q）——两批结果无法交叉校验；
  ②refusal=1.0 与"ASR 0.143"并存（当前代码 refused→"I don't know"→ASR 应为 0，
  旧代码路径与现标量不一致——**旧结果不可直接引用，必须重跑**）③重跑需 GPU ~数小时/实验。

## 12. 当前最严重的 7 个实验问题（按严重度）

1. **Refusal=100%（毁灭性）**：攻击密度（~4/5 投毒）+ S0 整联盟剔除 → 一切清空→ 全拒答。
   ASR 低是拒答换来的，utility 全丢。**必须重构选择器（Exp-2）。**
2. **split 不公平/泄漏**：baselines 100q vs ours 70q；按插入序划分；无 clean 集；变体池泄漏风险。
3. **CCI 是代理**：论文写 token-distribution JSD，实现是答案支持分数——未经验证近似。
4. **baseline 太弱**：仅 relevance/dup 过滤，无 LOO/发布防御/旧版。
5. **adaptive 轮次病态**：round2 逃脱率 1.00，且从不在生成 ASR 上验证；无 unseen held-out。
6. **单模型**：仅 Qwen2.5-7B；无跨 LLM 泛化。
7. **无统计**：无 paired bootstrap / 置信区间；表注无 n、无分布信息。

## 13. 可复用代码（直接复用/小改）

- `retrieval/{embedder,vector_store,keyword,fusion,kb}.py` —— 检索链稳定，直接用
- `relation/{nli,extractor,llm_judge,types}.py` —— 图边构建（含数值冲突启发式），直接用
- `graph/build.py` —— ClaimGraph + 联盟发现 + signed_semantic
- `coalition/{analyze,trust}.py` —— 结构特征与代理 CCI（保留，供 Exact CCI 对比）
- `defense/risk.py` —— RuleRisk / LearnedRisk（需补持久化进 protocol 结果目录 + 按 split 训练）
- `llm/*`、`config.py` —— 直接复用（K8 口径）
- `scripts/` 中的 clean_str/answer_match/_auc —— 复用并集中到 protocol 公共库
- 全部数据/索引/模型 —— 零重建

## 14. 必须重构

- **实验协议层**（本次建设核心）：`experiments/final_protocol/` 统一入口+冻结 split+schema 化结果
- **安全上下文选择**：S0/S1/S2 三策略 + val 约束调参（refusal≤10%, clean 掉点≤5%）+ 逐 query
  主状态标注（attack_success/correct/refusal/other 互斥）
- **评估口径**：baselines 与 ours 同一 test 池；生成只跑一圈；结果写 predictions.jsonl
- **Exact CCI 工具**：Qwen logits 探针 + teacher-forced 对齐 + JSD
- **自适应流程**：白盒攻击 = 以 detector 为信号的变体族 × 轮次；held-out unseen 变体族；
  输出攻防曲线；**最终检测器为交付物**
- **跨 LLM 基建**：hf 后端支持任意模型（已具备）；补 Llama-3.1-8B + 第三模型下载（~20G, 磁盘 OK）

---

## 附：文档与实现不一致（审计即发现，Phase 1 起以代码为准）

| 论文/模板表述 | 真实实现 |
|---|---|
| M3 “Query-conditioned Claim Extraction”（LLM 抽取） | 220-char 截断（无 LLM） |
| M6 CCI = 生成 token JSD | 答案支持代理（无生成探针） |
| M7 “hierarchical set-aware risk learner” | 单层小 MLP 6 特征（无分层/无 set attention——审阅意见已要求克制，此为实现略欠“分层“表述） |
| M8 “校准于验证集的风险预算“ | 无 val 校准，τ 全网格拒答 1.0 |
| Table 2/3 口径 | baseline 100q vs ours 70q——必须统一后重生成 |
