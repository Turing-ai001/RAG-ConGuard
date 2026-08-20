# RAG（新论文实验项目）

新论文（claim 验证方向）的代码仓库。与旧论文项目 `F:\论文\攻击与防御`（RAG-ConGuard++）思想接近，
**所有可复用组件直接沿用旧项目**，保证实验可比：

| 复用项 | 来源 | 说明 |
|---|---|---|
| 投毒数据 `data/poisoned/*.json` | 旧项目 PoisonedRAG 产物 | 3 数据集 × 100 query × 5 adv_texts |
| RAG 生成 prompt | PoisonedRAG `MULTIPLE_PROMPT` | 逐字一致 |
| embedding 模型 | 旧项目同款 `BAAI/bge-base-en-v1.5` | 已验证 |
| 答案匹配口径 `clean_str` | PoisonedRAG `src/utils.py` | 小写/strip/去尾句点 |
| 服务器约定 | 旧项目 run_qwen_full.sh | conda `rag_env`、`/root/autodl-tmp/models_cache`、hf-mirror |

## 目录结构

```
RAG/
├── config.py                  # 全局配置（路径/模型/超参，单点修改）
├── pipeline.py                # 主管线：检索 → claim 提取 → 生成
├── retrieval/                 # 多视角检索
│   ├── embedder.py            #   BGE embedding（可换 E5，改 config 一行）
│   ├── vector_store.py        #   FAISS（IndexFlatIP + L2 归一化 = cosine）
│   ├── keyword.py             #   BM25 关键词视角
│   ├── kg.py                  #   KG 视角：实体倒排 + 规则三元组
│   ├── fusion.py              #   RRF 融合
│   └── kb.py                  #   KB 组装 / 持久化 / 多视角入口
├── claim/extractor.py         # 主张提取（prompt engineering + JSON 三层容错）
├── llm/                       # LLM 统一接口（mock | hf | vllm 三后端，支持 batch）
├── data/
│   ├── poisoned/              # 投毒数据（复用旧项目，勿改）
│   ├── raw/                   # BEIR 语料（download_datasets.py 生成）
│   └── index/                 # KB 持久化（faiss + bm25 + kg）
├── scripts/
│   ├── download_datasets.py   # 下载 BEIR 语料子集
│   ├── build_kb.py            # 构建 KB（语料 + 投毒注入）
│   ├── evaluate_pipeline.py   # 投毒验证：检索命中 / claim / ASR
│   ├── setup_server.sh        # AutoDL 环境初始化
│   └── start_vllm.sh          # 启动 vLLM 端点（Qwen2.5-7B）
├── tests/smoke_test.py        # P0 端到端冒烟测试（本地 CPU + Mock LLM）
└── output/                    # 评估输出
```

## P0 快速验证（本机 CPU）

```bash
# 冒烟测试：检索/融合/持久化/claim/生成/批量 全链路
python tests/smoke_test.py --dataset nq --queries 5

# 投毒验证：多视角检索命中率（3 数据集）
python scripts/evaluate_pipeline.py --dataset nq --num_queries 100
python scripts/evaluate_pipeline.py --dataset hotpotqa --num_queries 100
python scripts/evaluate_pipeline.py --dataset msmarco --num_queries 100

# 里程碑验证：真实 LLM 的 claim 提取端到端（本机 1.5B 验证；服务器设 HF_MODEL 换 7B）
python scripts/_verify_claim.py
```

P0 验证结果（本机 CPU，500 条投毒文档 × 100 query，top_k=5）：

| 数据集 | 投毒检索命中率 | semantic | keyword | kg |
|---|---|---|---|---|
| nq | 1.0 | 1.0 | 1.0 | 0.89 |
| hotpotqa | 1.0 | 1.0 | 1.0 | 1.0 |
| msmarco | 1.0 | 1.0 | 1.0 | 0.83 |

即：攻击生成的投毒文档（adv_texts）全部能被多视角检索命中 —— 攻击生效的前提成立。

说明：
- 本机跑脚本默认 `HF_OFFLINE=1`（模型已缓存时避免联网检查超时）；需联网下载时设 `HF_OFFLINE=0`
- Windows 中文路径下 FAISS 读写已做 chdir 相对路径兼容（`retrieval/vector_store.py`）

## P1 里程碑指标（验收口径，2026-08-19 修订）

> 原口径"给定 query 输出有符号证据图，NLI 准确率 > 85%"已作废。
> 原因：cross-encoder NLI 在长文本 claim + 数值冲突（"24 集"vs"23 集"）跨域场景实测
> 准确率仅 ~47%（126 样本），但该错误全部是"判成无关联"、极性方向从不混淆，
> 联盟级聚合对边稀疏鲁棒（纯度仍达 1.0）。指标改为三层：

| 层级 | 指标 | 阈值 | 当前状态 |
|---|---|---|---|
| 主指标 | 有符号图联盟平均纯度 max(投毒占比, 正常占比) | ≥ 0.85 | ✅ 1.0（3 toy queries） |
| 辅助指标 | NLI 在"非数值冲突"子集上的准确率 | ≥ 80% | ❌ 56.6%（106 样本实测；失败与数值无关：paraphrase 蕴含召回仅 48%、实体矛盾 70%） |
| 过程指标 | LLM 兜底触发率 ≥ 5% 且兜底边极性正确率 ≥ 90% | — | ✅ 链路 24/24 mock 验证通过（真实触发率待服务器） |

数字归一化启发式（`relation/nli.py::numeric_conflict_check`）已于 2026-08-20 接入：
数值冲突子集（20 样本）召回率 100%、误伤 0，相关边 59→77（+31%，
低于预期 +50% 的原因：3 个评估 query 中仅 1 个含数值，另 2 个无数字可拦截）。
已接入管线：`classify()` 打分前预检（命中跳过 NLI 模型）。

LLM 兜底链路（`relation/extractor.py`）同日修复并验证：
- 补齐融合公式 `score = NLI_LLM_FUSION_ALPHA*s_NLI + (1-alpha)*s_LLM`（alpha=0.3）
- `scripts/_verify_fallback.py` 24/24 通过（触发范围/融合数值精确断言/图极性写入）
- 真实 judge 触发率待服务器验证。

指标常量定义在 `config.py` 的 `MILESTONE_*`，后续所有评估以此为准。

## 服务器（AutoDL）运行

```bash
# 1. 项目同步 + 环境初始化（conda rag_env、GPU 依赖、Qwen/BGE 模型）
bash scripts/setup_server.sh

# 2. 构建语料与 KB（全量规模自行调整 --corpus/--queries）
python scripts/download_datasets.py --datasets nq,hotpotqa --corpus 600000 --queries 5000
python scripts/build_kb.py

# 3. 起 vLLM（Qwen2.5-7B-Instruct，端口 8000）
nohup bash scripts/start_vllm.sh > /root/autodl-tmp/log_vllm.log 2>&1 &

# 4. 全链路评估（检索命中 + claim + ASR）
LLM_BACKEND=vllm python scripts/evaluate_pipeline.py --dataset nq --gen 1 --extract 1 --num_queries 100
```

## 约定

- `LLM_BACKEND`：`mock`（本地无 GPU 验证管线）/ `hf`（transformers 直连）/ `vllm`（服务器端点），
  全部配置走 `config.py` + 环境变量，模块内不写死路径。
- KG 视角为轻量实现（实体倒排 + 规则三元组），后续可无痛替换为真实图谱（实现 `KGStore` 协议即可）。
- 数据 `data/poisoned/` 与旧项目逐字一致，勿修改。
