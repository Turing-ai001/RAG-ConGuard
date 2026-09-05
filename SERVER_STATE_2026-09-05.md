# 服务器状态记录（2026-09-05 审计前冻结）

> 记录主机：`ssh -p 44793 root@connect.nmb1.seetacloud.com`（AutoDL，密码见本机凭据/会话，勿入 git）
> 记录时点：2026-09-05 12:20 (UTC+8)。本文件与 `AUDIT_2026-09-05.md` 配套。

## 1. 运行状态

- **没有任何实验在运行**。`ps aux` 仅有 jupyter-lab / tensorboard / supervisord / autopanel 等系统进程；GPU 0%（RTX 4090 24G 空闲）。
- 最近一次实验活动：2026-08-25 22:41（p6_adaptive_unseen_v3.log）。此后代码与结果目录无修改。
- load average 8.24 为容器宿主侧数值，与实验无关。

## 2. 代码版本

- 本地 git 快照：`69c6cd68b143b95c66dbe9fd1643a9526865fe73`（2026-09-05 12:10, 作者 yangjs），工作区干净。
- 服务器非 git 仓库；对本地跟踪的 260 个文件逐一 md5 比对：
  - **IDENTICAL: 109 / DIFFER: 0 / 服务器不存在: 151**（论文 docx/md、figure、latex_test、以及结果文件——后者在服务器上位于 `results/p1|p6` 子目录，见 §4 映射）。
- 服务器代码文件时间戳均 ≤ 2026-08-25 17:54（llm/hf_backend.py，新增 probe_logits），8-25 后无改动 → **服务器代码 = 本地快照 69c6cd6 的代码**。

## 3. 环境与数据

| 项 | 值 |
|---|---|
| GPU | RTX 4090 24G（CUDA 可用） |
| conda | `rag`（/root/miniconda3/envs/rag，torch 2.5.1+cu124） |
| 项目 | /root/autodl-tmp/RAG |
| 模型缓存 | bge-base-en-v1.5 / Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct / Mistral-7B-Instruct-v0.3 / cross-encoder-nli-deberta-v3-base（全部在 /root/autodl-tmp/models_cache/） |
| 语料 | data/raw/{nq,hotpotqa,msmarco}_{corpus,queries}.jsonl（600k corpus；832M） |
| KB 索引 | data/index/ 6.9G（2026-08-24 构建，冻结不重建） |
| 投毒 | data/poisoned/{ds}.json（100 query × 5 adv_texts / 数据集） |

## 4. 数据划分（冻结，splits/*.json，only-once 生成，两侧一致）

执行：`scripts/make_splits.py`（seed 20260825：投毒 id 排序→Random(SEED).sample→切片；clean 用 SEED+1）。

- 投毒查询：**train 60 / val 20 / test 20**（每数据集各 100 → 共 180/60/60）
- 干净查询：**clean_trn 200 / clean_val 100 / clean_test 300**（每数据集；clean_trn 仅作风险学习器负例池）
- 交集检查：train/val/test 与 clean_trn/clean_val/clean_test 相互不相交（脚本内 assert + 数字复核 nq=60/20/20 等 ✓）

## 5. 输出目录映射（服务器 ↔ 本地）

| 服务器路径 | 本地路径 | 内容 |
|---|---|---|
| results/p1/ | results/（平铺） | P1 链全部：features_{ds}_{split}.jsonl×18、risk_learner.pkl、tune.json、tune_baselines.json、eval_test_*/eval_clean_*/metric+pred、exact_cci_*（val H8/H16/H32 + train H16）、eval_loo_final.json、MAIN_DONE 等标记 |
| results/p1semsem/ | 无（本地未同步） | signed_semantic 图消融：features×12 + tune.json + risk_learner.pkl（仅特征与调参，未见 eval 输出） |
| results/p6/ | results/ | P6 自适应：adaptive_curve.json（v3, 2026-08-25 22:33）、adaptive_unseen.json（v3, 22:41）、risk_r1/r2.pkl、P6V2/V3_DONE |

## 6. 已执行运行（均在 2026-08-25，无挂起）

| 阶段 | 脚本/命令 | 时点 | 产物 |
|---|---|---|---|
| P1 链 | scripts/run_p1_chain.sh | ~13:37 | features 9 文件+clean_val 特征+risk_learner+早期 test B0,G0 |
| P1 调参 | fp.tune（v1.2 版） | 13:37 | results/p1/tune.json → chosen S2 λ=0.2（ASR 0.633/ref 0.100，val） |
| 基线调参 | fp.tune_baselines | 19:08 | tune_baselines.json（扫描，**无 chosen 字段**） |
| 主结果 | run_main_eval.sh | 19:27 | eval_test_B0_B1_B2_B4_B5_B6_G0_G1（主表） |
| clean | run_clean_eval.sh | 17:12/17:39 | eval_clean_val|test_B0_G0_G2 |
| exact CCI | fp.exact_run | 17:44–19:36 | exact_cci_val_{H8,H16,H32}, exact_cci_train_H16 |
| P6 自适应 | fp.adaptive（v1/v2/v3） | 20:22–22:41 | adaptive_curve.json v3、adaptive_unseen.json v3 |
| 跨 LLM | run_cross_llm.sh | 21:24–21:46 | eval_*_{Instruct|v0.3}（tag 命名） |
| Rule 消融 | run_ablation.sh | 20:49 | eval_test_G0_G1（--risk-kind rule） |
| E8 | scripts/eval_loo_final.py | — | eval_loo_final.json |

### 6.1 执行参数事实（6.2 的发现，来源 = 三大 run 脚本原文 + 复算）

- 主表（run_main_eval.sh）：`--methods-params '{"B1":{"tau_rel":0.30},"B2":{"tau_dup":0.7},"B6":{"tau_loo":0.5},"G0":{"selector":"s0","tau":0.05},"G1":{"selector":"s2","lam":0.2}}'`
  - **G1 合并 METHODS 默认（selector=s1, tau=0.25）后 = S2@λ0.2+τ0.25**（tau 残留，复算 avg_removed=1.667 与 predictions 一致；tune 扫描的 S2 是 τ=0.0）
  - G0 = S0@τ0.05（复算 3.700 一致）；B2 τ_dup=0.7、B6 τ_loo=0.5（矩阵覆盖默认值）
- clean（run_clean_eval.sh）：`{"G0":{"tau":0.25},"G2":{"lam":0.2}}` → clean 的 G2 = S2@λ0.2+**τ=0.0**（G2 无默认 tau）；G0(clean)=S0@τ0.25（**与主表 G0 的 τ0.05 不同**）
- 跨 LLM（run_cross_llm.sh）：与主表相同 G0/G1 覆盖（含 τ 残留）
- Rule 消融（run_ablation.sh）：`--risk-kind rule` + 同 G0/G1 覆盖（Rule G1 同样 τ0.25 残留）

### 6.2 config.json 局限性（直接影响溯源）

`fp.runner.stage_eval` 落盘的 config.json 只含 protocol/split/methods/datasets/seed，**不含方法参数**（--methods-params 覆盖值不落盘）。方法参数仅存在于 shell 脚本与 METHODS 默认值合并逻辑中；predictions.jsonl 的 `model` 字段**硬编码 "qwen2.5-7b-instruct"**（不随 HF_MODEL 变化，段）

## 7. 新实验约定（自 2026-09-05 起）

- 一切新运行使用**新的独立目录**：`results/p1_<audit|补跑>_<日期>/`（fp.runner --run-dir 直接指定），禁止覆盖 results/p1（其内容 = FTB 结果基线，已入 git 69c6cd6）。
- 服务器与本地同步：本地为源（scripts/sync_to_server.py, 需 env SSHPW）；同步后 `python -m py_compile` 校验。

## 8. TeX 版本记录（当前 TeX）

- `latex_test/` 仅含 **VSCode LaTeX Workshop 编译环境测试**（ctexart + xelatex；test.tex 2026-09-02 10:58；sigconf.tex 为 ACM 模板样例），**非论文投稿稿**。
- 已随 git 快照 69c6cd6 入库（test.tex/test.pdf/test.aux/test.out/test.toc/sigconf.tex/sigconf.pdf/comment.cut/…）。
- 论文正式稿 = Word 体系（RAG_论文框架_初稿_v4.docx + paper_draft_cards_full.json + scripts/build_paper_draft.py），同快照入库。
- 无独立 main.tex 投稿版——**Table 1-4 转 LaTeX 仍为未完成项**（记忆 card: paper-writeup-state 下一步 4）。

## 9. 待核查（本会话发现、未定案）

- `results/p1/eval_clean_val_B0_G1/`（服务器 mtime 2026-08-25 20:47）：B0 clean refusal=0.0767、答案均为长文本（avg 190 字符）；与当前代码重跑结果（"I don't know." 短答、refusal 0.27）**不一致**。未找到其启动命令/日志（无对应 log 文件）。其 benign_mis_filter=0.0713 与跨 LLM 运行一致（mis-filter 与生成无关 → 可信），但 refusal 行为可疑 → **该目录整组数字暂不采用；报告 §3/§8 所用的 0.27/0.286 与当前代码一致 ✓**。
