# 论文实验数据表（Table 1-4, 初稿配套）

> 数字均来自 output/eval_pipeline/ 下回收的评估 JSON；设置：NQ/HotpotQA/MS MARCO × 600k 语料 × 100 评价查询/数据集 × 每查询 5 篇投毒文档注入；检索 BGE-base-en-v1.5(FAISS)+BM25（本次评估 KG 视角关闭）；生成 Qwen2.5-7B-Instruct(bf16, T=0)。

## Table 1. 检索阶段攻击面与多视角命中率（受攻无防御）

| Dataset | Retrieval ASR @top-5 | Avg poisoned docs in top-5 | Dense view hit | Lexical (BM25) view hit |
|---|---|---|---|---|
| NQ | 0.99 | 3.88 | 0.99 | 0.94 |
| HotpotQA | 1.00 | 4.89 | 1.00 | 1.00 |
| MS MARCO | 0.97 | 3.12 | 0.98 | 0.86 |

*注：Retrieval ASR = 查询命中率（top-5 中出现投毒文档的查询比例）；该表说明攻击材料进入 prompt 的程度，是防御生效的前提。*

## Table 2. 生成阶段攻击成功率（Generation ASR）

| Dataset | Vanilla RAG | Relevance filter | Duplicate filter | **RAG-ConGuard (τ=0.25)** |
|---|---|---|---|---|
| NQ | 0.86 | 0.86 | 0.76 | 0.14 |
| HotpotQA | 0.99 | 0.99 | 0.93 | 0.14 |
| MS MARCO | 0.82 | 0.82 | 0.70 | 0.14 |

*注：RAG-ConGuard 数值为跨数据集聚合的 held-out 评估集（70 查询，30q/数据集训练、70 查询验证切分）；Vanilla/基线为 100 查询/数据集。Relevance filter 阈值 τ_rel=0.30（保留全部 5 篇）；Duplicate filter 阈值 τ_dup=0.85（平均保留 2.3/1.2/2.3 篇）。*

## Table 3. 生成准确率与防御代价（受攻设置）

| Dataset | Vanilla (attacked) | Relevance filter | Duplicate filter | RAG-ConGuard (τ=0.25) | Refusal rate (ConGuard) |
|---|---|---|---|---|---|
| NQ | 0.18 | 0.18 | 0.31 | 0.06¹ | 1.00 |
| HotpotQA | 0.03 | 0.03 | 0.06 | 0.06¹ | 1.00 |
| MS MARCO | 0.22 | 0.22 | 0.38 | 0.06¹ | 1.00 |

*¹ ConGuard 数值为聚合 held-out 集（正确率 0.057）。注意：当前实现的安全子集在 τ 全范围上产生全额拒答（refusal=1.0），即防御以"拒答所有查询"换取 ASR 大幅下降；高准确率的恢复留待风险预算/细粒度 τ 工作——详见 §4.6 讨论。*

## Table 4. 冗余投毒下逐文档 LOO vs 联盟级 CCI（E8 诊断实验，审阅意见硬性要求）

> 口径：在评估集 120 个投毒组上重放同一影响公式（删除 claim/联盟后 trust 传播+答案支持重算，max_a 相对影响）。

| Poisoned group size | # groups | Avg. per-document LOO | Avg. coalitional removal | Ratio |
|---|---|---|---|---|
| 2 docs | 29 | 0.159 | 0.161 | 1.01x |
| 3 docs | 41 | 0.372 | 0.593 | **1.60x** |
| 4 docs | 51 | 0.463 | 0.918 | **1.98x** |

*注：投毒组规模增大时，单篇 LOO 被剩余成员稀释（慢增 0.159→0.463），联盟级删除控制力快速放大（0.161→0.918）。这就是"逐文档归因在协同投毒下失效"的直接数量证据。*
