# RAG-ConGuard 论文初稿（英文，按模仿框架填入）

RAG-ConGuard: Coalition-Aware Counterfactual Defense

论文模仿框架（大师兄三步法）· 主要参考文献: ShieldRAG (AAAI-26) · 目标: 三四区SCI 双栏IEEE风格

使用说明

① 按卡片编号顺序填写：先在 ✏️ 写作区写中文草稿 → 再按 💬 Reference pattern 套英文句式改写 → 每张卡片标注了篇幅目标（~words 数）。

② 黄色虚线框是图表占位：编号、展示内容、参考来源（抄哪张图）、作图建议已给出，实验跑完后替换为真实图表。

③ 关键卡片带 ★：I5 关键挑战、M4/M6/M7/M8/M9 核心机制、E8 对照实验、E5-E11 的「图→趋势→机制」三步是三四区审稿底线，务必按序写出。

④ 全部卡片填完 = 初稿完成；回到对话可触发 Phase 3 降重（AI 降重 + 人工 2-3 遍），提交前通用术语检查一遍。

推荐 Index Terms：Retrieval-Augmented Generation; Knowledge Poisoning Defense; Signed Evidence Graph; Coalitional Counterfactual Influence; Adversarial Training; Safe Context Selection

Title（三选一，投稿前可按目标期刊长度微调）

「T1」A型 · 名字+问题域（推荐）

RAG-ConGuard: Coalition-Aware Counterfactual Defense against Coordinated Knowledge Poisoning in Retrieval-Augmented Generation

── | 句式来源: ShieldRAG 命名式 Name: Gerund from Threat

「T2」B型 · Safeguarding 式

Safeguarding Retrieval-Augmented Generation from Coalitional Poisoning: Evidence-Graph-Guided Counterfactual Influence

── | 句式来源: ShieldRAG 副标题式

「T3」C型 · 悬念式

Uncovering Poisoning Coalitions: A Signed Evidence Graph and Coalitional Counterfactual Defense for RAG

── | 若目标期刊喜悬念可用

Abstract（单栏 · 打印样式 9pt）

「A1」背景现状

在 RAG 系统广泛集成开放知识库的背景下，指出开放来源未经过筛选、必然存在不可信内容。

第 1 句：现象 + 固有风险。

💬 Reference pattern: Open knowledge bases (e.g., websites) are widely adopted in Retrieval-Augmented Generation (RAG) systems to provide supplementary knowledge. However, such sources inevitably contain biased or harmful content, and incorporating these untrusted contents into the RAG process introduces significant safety risks [cite].

Open knowledge sources, such as publicly available web corpora, are now widely integrated into Retrieval-Augmented Generation (RAG) systems to provide supplementary context. However, such sources are uncurated and inevitably contain biased or harmful content; incorporating these untrusted contents into the RAG pipeline introduces significant safety risks [Gao et al. 2023; Ni et al. 2025].

── P-Abstract-1 | ~45 words | 引用: RAG 调研 1-2 条

「A2」协同投毒威胁

第 2 句：投毒攻击放大风险，且已从单篇注入演变为【多篇互相支持、共同控制生成】的协同形式。

要点："移除任何一篇仍有剩余控制力" 一句带出联盟概念。

💬 Reference pattern: Recent studies have shown that this vulnerability can be further amplified by adversarial poisoning attacks specifically targeting the knowledge sources. More sophisticated attacks jointly inject multiple mutually-supporting passages so that collectively they dominate the generated output, and removing any single one of them leaves the remaining ones fully in control.

Recent studies have shown that this vulnerability can be further amplified by poisoning attacks that deliberately inject adversarial passages into the knowledge source. More sophisticated attacks jointly inject multiple mutually-supporting passages so that, collectively, they dominate the generated output; removing any single one of them leaves the remaining ones fully in control.

── P-Abstract-2 | ~50 words | 引用: 投毒攻击 1-2 条

「A3」现有防御局限

第 3 句：现有防御大多逐篇独立判断（困惑度/嵌入距离/文本相似度等人工统计特征），忽略候选文档间的结构与冗余关系，且在自适应攻击下退化。

留给你的一句话要能链接到 "Unsupported Influence" 缺位。

💬 Reference pattern: Most existing defenses examine each retrieved document independently using hand-crafted features such as perplexity, embedding distances, and textual similarity, failing to capture the structural relations among candidate documents; they also degrade under adaptive attacks.

Most existing defenses examine each retrieved document independently, using hand-crafted features such as perplexity, embedding distances, and textual similarity. They therefore fail to capture the structural and redundant relations among candidate documents, and they degrade under adaptive attacks.

── P-Abstract-3 | ~40 words | 引用: 弱防御线 1-2 条

「A4」提出框架

第 4 句：提出 RAG-ConGuard——一个对检索证据集合做结构化风险推理的联盟感知反事实防御框架。

💬 Reference pattern: In this paper, we propose RAG-ConGuard, a coalition-aware counterfactual defense framework that reasons over the structure of the retrieved evidence set through multi-view retrieval stability, a query-conditioned signed evidence graph, and coalition-level generation control.

In this paper, we propose RAG-ConGuard, a coalition-aware counterfactual defense framework that reasons over the structure of the retrieved evidence set through multi-view retrieval, a query-conditioned signed evidence graph, and coalition-level influence attribution.

── P-Abstract-4 | ~40 words | 无新引用

「A5」核心机制

第 5 句（本轮最重要）：先在证据图上发现联盟，再【整体删除联盟】测反事实生成影响（JS 散度），从而识别"缺少独立支持却不成比例控制生成"的投毒联盟。

强调：无需事先知道攻击目标。

💬 Reference pattern: The core idea is to first discover evidence coalitions on the signed evidence graph and then measure their coalitional counterfactual influence: the Jensen–Shannon divergence of the generation distribution after removing the whole coalition. This captures the disproportionate, under-supported control of coordinated poisoned documents without knowing the attack target in advance.

The core idea is to first discover evidence coalitions on the signed evidence graph and then measure their coalitional counterfactual influence: the change in a trust-weighted answer-support score after removal of the whole coalition. This captures the disproportionate, under-supported control of a coordinated poisoned group without knowing the attack target in advance.

── P-Abstract-5 | ~55 words | 无新引用

「A6」学习与生成

第 6 句：分层集合风险学习器（文档/联盟/查询三级风险）+ 风险约束安全上下文选择；再一句给出 ADSEA-C 联盟式自适应攻击与攻击者在环训练。

💬 Reference pattern: To this end, a hierarchical set-aware risk learner and a risk-constrained safe context selector jointly identify dangerous coalitions and construct safe evidence subsets, while a coalition-aware adaptive attack (ADSEA-C) drives adversary-in-the-loop training.

A hierarchical set-aware risk learner and a risk-constrained safe context selector jointly identify dangerous coalitions and construct safe evidence subsets, and a coalition-aware adaptive attack drives adversary-in-the-loop training of the learner.

── P-Abstract-6 | ~40 words | 无新引用

「A7」实验设置与规模

第 7 句：在 [N] 个真实世界数据集、[L] 个主流行 LLM、[K] 种投毒攻击策略（含本文 ADSEA-C 与旧版基线）下评估，报告检索/生成 ASR、Recall@k 与生成准确率。

若实验已定稿，直接填数字。

💬 Reference pattern: We evaluate RAG-ConGuard on [N] real-world datasets with [L] widely-used LLMs under [K] poisoning attack strategies, reporting retrieval and generation attack success rates (ASR), Recall@k, and generation accuracy.

We evaluate RAG-ConGuard on three real-world benchmarks (NQ, HotpotQA, and MS MARCO) with a widely-used open LLM (Qwen2.5-7B-Instruct) under targeted poisoning and coalition-wise adaptive attack variants, reporting retrieval and generation attack success rates (ASR) and generation accuracy.

── P-Abstract-7 | ~40 words | 无新引用

「A8」结果与意义

第 8 句：结果句（ASR 降至近零、精度/召回恢复至攻击前水平）+ 意义句（联盟级结构化推理是可信 RAG 的关键组件）。

可选：代码仓库。

💬 Reference pattern: Experimental results show that RAG-ConGuard substantially improves robustness, suppressing ASR to near-zero levels while maintaining competitive accuracy and retrieval utility, demonstrating that coalition-level structural reasoning can serve as a critical component for building trustworthy RAG systems.

💬 Reference pattern: Code is available at [link].

Experimental results show that RAG-ConGuard substantially improves robustness, suppressing generation ASR from 0.86-0.99 to 0.14 while retaining competitive utility, and that similarity- or relevance-based single-document filters make no measurable gain under the same setting, demonstrating that coalition-level structural reasoning is a critical component for trustworthy RAG systems.

── P-Abstract-8 | ~50 words | 无新引用

1. Introduction

「I1」RAG 普及与开放知识库价值

三段式第 1 段（2-3 句）：①RAG 因"可访问最新/领域知识"成为主流；②开放来源是主要知识供给；③给 2-3 篇综述引用。

💬 Reference pattern: Open knowledge sources, such as publicly available websites, are now widely integrated into Retrieval-Augmented Generation (RAG) systems to supply supplementary context. This integration allows LLMs to access up-to-date, domain-specific knowledge [Gao et al. 2023; Ni et al. 2025], which is essential for real-world applications.

Large language models (LLMs) cannot be trained on all knowledge that users may ask about, and Retrieval-Augmented Generation (RAG) has therefore become the dominant architecture for injecting up-to-date and domain-specific knowledge at inference time. Most deployed RAG systems draw their supplementary context from open knowledge sources, such as public web corpora and community repositories, because these sources are abundant, zero-maintenance, and broadly cover real-world queries [Gao et al. 2023; Ni et al. 2025]. This integration allows LLMs to answer questions that lie beyond their parametric knowledge, which is essential for real-world applications such as open-domain question answering and long-horizon assistants.

── P-Intro-1 | ~110 words | 引用建议: [Gao et al. 2023], [Ni et al. 2025], [Huang et al. 2025]

「I2」开放知识库固有风险（Fig.1）

①未筛选来源必含偏差/有害内容；②有害段落进入 prompt 会触发三类失败模式（对应 Fig.1 情景图——请画【协同投毒联盟以"互相支持"伪装成多份独立证据，误导生成】）；③强调对用户可信度的侵蚀。

💬 Reference pattern: However, these open sources inevitably contain biased or harmful information due to their uncurated nature, and integrating such untrusted content into the RAG pipeline introduces novel safety risks [cite]. As illustrated in Fig. 1, when a coalition of mutually-supporting adversarial passages is incorporated into the prompt, it can trigger typical failure modes: deceptive support of wrong answers, degradation of response quality, and reproduction of harmful statements.

However, open knowledge sources are uncurated, and they inevitably contain biased, misleading, or harmful information. Integrating such untrusted content into the RAG pipeline introduces novel safety risks [Zhong et al. 2023]: when an adversarial passage is retrieved and included in the prompt, the model can be steered to echo the injected assertion. In the coordinated setting studied in this work, a group of mutually-supporting passages is crafted so that, even though no single one is dominant, together they dominate the evidence set and mislead generation. Such failures erode user trust in the retrieved answers.

── P-Intro-2 | ~130 words | 引用建议: 安全性风险 1-2 条

「I3」投毒攻击的演进

①回顾攻击演进（按时间+能力递进，勿流水账）：Zhong 2023 注入对抗段落污染稠密检索 → One Shot Dominance 2025 低成本主导检索 → RIPRAG 2026 强化学习黑盒攻击；②落脚：攻击者已能构造"互相支持的文档组"，使威胁从单文档走向联盟；③一句"威胁面扩大"。

💬 Reference pattern: Unfortunately, recent studies have demonstrated that the vulnerability of RAG systems can be further exacerbated by poisoning attacks [Zhong et al. 2023]. These attacks involve the deliberate design and injection of adversarial passages tailored to exploit common retrieval patterns, and emerging attacks coordinate several mutually-supporting passages or even operate in black-box settings [Chang et al. 2025; Xi et al. 2026]. Collectively, such attacks significantly amplify the inherent safety risks of uncurated knowledge sources and broaden the overall threat surface of RAG systems.

Unfortunately, recent studies have demonstrated that the vulnerability of RAG systems can be further exacerbated by poisoning attacks [Zhong et al. 2023]. Emerging attacks coordinate several mutually-supporting passages so that the threat moves from a single injected document to a coalition [Chang et al. 2025; Xi et al. 2026], for example by boosting the retrieval likelihood of a group of near-duplicate adversarial passages. Collectively, such attacks significantly amplify the inherent safety risks of uncurated knowledge sources and broaden the threat surface of RAG systems.

── P-Intro-3 | ~140 words | 引用建议: [Zhong et al. 2023], [One Shot Dominance 2025], [RIPRAG 2026], 可选 [PoisonedRAG], [BadRAG]

「I4」现有防御的不足

按四类不足补刀：①手工特征过滤（困惑度/嵌入距离，RobustRAG 类高延迟或失真）；②聚类+重排（TrustRAG 在自适应攻击下退化）；③NLI 一致性（ReliabilityRAG 逐文章二分、无联盟概念）；④归因/注意力（AttnTrace、Who Taught the Lie 依赖特定信号）。

共同病根：**逐文档独立判断 + 静态阈值**，看不到"联盟级冗余控制"。

💬 Reference pattern: Most existing approaches evaluate retrieved content document by document using hand-crafted features, such as likelihood, embedding distances, and perplexity [Zhong et al. 2023; Xiang et al. 2024], to filter low-quality retrieval content. More recent defenses resort to clustering-based filters [Zhou et al. 2025], NLI-based consistency graphs [Shen et al. 2025], or attribution signals [Wang et al. 2026], yet they still handle each document independently and degrade under adaptive and coordinated attacks.

Most existing defenses evaluate retrieved content document by document, relying on hand-crafted features such as likelihood, embedding distances, and perplexity to filter low-quality content [Zhong et al. 2023; Xiang et al. 2024]. More recent approaches resort to clustering-based filters [Zhou et al. 2025], NLI-based consistency reasoning [Shen et al. 2025], or attribution signals [Wang et al. 2026]; yet they still process each document independently, rely on static thresholds, and are not designed for coordinated groups of passages. Because the signal of a coordinated attack resides in the structural and redundant relations among documents rather than in any individual document, these defenses cannot capture coalition-level control.

── P-Intro-4 | ~160 words | 引用建议: [RobustRAG], [TrustRAG], [ReliabilityRAG], [AttnTrace], [Who Taught the Lie], [Push and Pull]

「I5」关键挑战 ★

①转折句：现有研究大多关注提升 RAG 的准确与效率，忽视此类安全问题；②灵魂句（必写）：如何在生成前识别"缺乏独立证据支持却对生成产生不成比例控制力的投毒联盟"，且不牺牲检索效用；③埋伏笔：高困惑度/高检索分 ≠ 恶意——权威证据也可能高影响（为 Unsupported Influence 假设做铺垫）。

💬 Reference pattern: Therefore, how to enhance the robustness of RAG systems against coordinated knowledge poisoning without compromising its performance remains a key challenge. Notably, a high-impact evidence group is not necessarily malicious: authoritative evidence supporting the correct answer may also exert strong influence, meaning naive impact-based criteria cannot serve as the risk indicator.

Therefore, how to enhance the robustness of RAG systems against coordinated knowledge poisoning without compromising retrieval and generation utility remains a key challenge. Notably, a high-impact evidence group is not necessarily malicious: authoritative evidence that supports the correct answer may also exert strong influence. Hence, neither high retrieval relevance nor high individual influence can serve as the risk indicator; a systematic criterion is needed that distinguishes under-supported, disproportionately controlling evidence groups from merely important ones.

── P-Intro-5 | ~120 words | 承接 I2-I4, 无新引用

「I6」本文提出 RAG-ConGuard

①一句总述：提出联盟感知反事实证据防御框架 RAG-ConGuard；②正式定义核心安全假设 **Unsupported Influence**（成功投毒=无独立支持的不成比例控制）；③七模块流水线一句话（多视角检索→主张提取→有符号证据图→联盟发现→联盟级 CCI→分层集合风险学习→安全子集构造）；④实验概览一句；⑤代码占位。

💬 Reference pattern: In this paper, we propose a coalition-aware counterfactual defense framework (RAG-ConGuard) that performs structured risk reasoning over the entire retrieved evidence set. The core safety hypothesis, termed Unsupported Influence, is that a successful coordinated poisoning attack manifests as an evidence coalition that exerts disproportionate control over generation while lacking independent supporting evidence. ... Experiments on [N] datasets, [L] LLMs, and [K] attack strategies (including our coalition-aware adaptive attack and an adversary-in-the-loop training scheme) demonstrate that RAG-ConGuard substantially enhances robustness while preserving utility.

In this paper, we propose RAG-ConGuard, a coalition-aware counterfactual defense framework that performs structured risk reasoning over the entire retrieved evidence set. The core safety hypothesis, termed Unsupported Influence, is that a successful coordinated poisoning attack manifests as an evidence coalition that exerts disproportionate control over the generation while lacking independent supporting evidence. Following this hypothesis, RAG-ConGuard operates as a pipeline of six stages: multi-view retrieval, query-conditioned claim extraction, signed evidence graph construction, coalition discovery, coalitional counterfactual influence attribution, and risk-constrained safe context selection; the risk learner is additionally strengthened by adversary-in-the-loop training. Experiments on three datasets with an open 7B LLM under targeted poisoning and its coalition-wise adaptive variants demonstrate that RAG-ConGuard substantially enhances robustness while preserving utility.

── P-Intro-6 | ~170 words | 无新引用

「I7」三贡献 bullets

三条贡献，每条 1-2 句，与 Methods 三大模块一一对应：①【首次】面向协同投毒，将查询相关证据图与联盟级反事实生成影响结合；②分层集合风险学习器 + 风险约束安全上下文选择；③ADSEA-C 联盟式自适应攻击 + 攻击者在环对抗训练。

（"首次"措辞采用你设计文档第 17 节的组合级声明，逐组件不称首次。）

💬 Reference pattern: • We propose RAG-ConGuard, the first framework that models coordinated knowledge poisoning by combining a query-conditioned signed evidence graph with coalitional counterfactual generation influence. • We introduce a hierarchical set-aware risk learner together with a risk-constrained safe context selection algorithm that constructs safe evidence subsets under a preset benign mis-filtering budget. • We design ADSEA-C, a coalition-aware adaptive attack, and perform adversary-in-the-loop training, showing that the resulting defense learns stable risk boundaries against evolving attacks.

The contributions of this paper are threefold. First, we propose RAG-ConGuard, which models coordinated knowledge poisoning as evidence coalitions on a query-conditioned signed evidence graph and measures their coalition-level counterfactual influence over a trust-weighted answer-support score, thereby identifying under-supported yet disproportionately controlling groups without knowing the attack target in advance. Second, we introduce a hierarchical set-aware risk learner together with a risk-constrained safe context selection algorithm that constructs safe evidence subsets by removing risky coalitions and refusing when no safe subset exists. Third, we design a coalition-aware adaptive attack and evaluate adversary-in-the-loop training, showing that the resulting defense learns to resist evolving evasion strategies that defeat static per-document filters.

── P-Intro-7 | ~150 words | 无新引用

2. Related Work

「RW1a」Related Work 1 · 针对检索阶段与全系统的攻击

按攻击对象分级：先稠密检索层（Zhong’23 注入对抗段落、Long’24 语法隐门），说明"少量注入即误导检索"；为下一段过渡。

💬 Reference pattern: Since dense retrieval forms the backbone of RAG systems by measuring query-knowledge relevance [Zhao et al. 2024], we first introduce attack methods on the dense retrieval model and then attacks on the whole RAG system. Recently, many studies demonstrate that attackers can manipulate retrievers to return crafted adversarial content: Zhong et al. (2023) show that even a small number of perturbed passages can mislead retrieval across domains, and Long et al. (2024) introduce a covert backdoor that leverages grammatical errors as hybrid triggers.

Since dense retrieval forms the backbone of RAG systems by measuring query-knowledge relevance [Zhao et al. 2024], we first introduce attack methods on the dense retrieval model and then on the whole RAG system. Many studies demonstrate that attackers can manipulate retrievers to return crafted adversarial content: Zhong et al. (2023) show that even a small number of perturbed passages can mislead retrieval across domains, and Long et al. (2024) introduce a covert backdoor that leverages grammatical errors as hybrid triggers.

── P-RW-1a | ~120 words | 引用建议: [Zhao et al. 2024], [Zhong et al. 2023], [Long et al. 2024]

「RW1b」Related Work 1 · 全系统与新兴协同/黑盒攻击

①全系统攻击线：BadRAG（Xue’24）检索后门诱导生成、Tan’24b 多阶段联合攻击、PoisonedRAG（Zou’24）语料级注入；②2025-26 新趋势：低成本/黑盒/协同（One Shot Dominance’25 每查询单文档主导输出、RIPRAG’26 RL 黑盒）；③总结句：威胁模型转变催生新防御需求（衔接 Methods 与 Related Work 2）。

💬 Reference pattern: Building on these vulnerabilities, more recent efforts target the entire RAG system by exploiting both retrieval and generation stages: Xue et al. (2024) create retrieval backdoors that affect LLM responses, Tan et al. (2024b) concatenate crafted segments to jointly poison retrieval and generation, and Zou et al. (2024) induce attacker-specified outputs through corpus-level injection. Notably, emerging attacks coordinate multiple passages or operate under black-box settings [Chang et al. 2025; Xi et al. 2026], which further complicates retrieval-time defense.

Building on these vulnerabilities, more recent efforts target the entire RAG system by exploiting both the retrieval and the generation stages: Xue et al. (2024) create retrieval backdoors that affect LLM responses, Tan et al. (2024b) concatenate crafted segments to jointly poison retrieval and generation, and Zou et al. (2024) induce attacker-specified outputs through corpus-level injection. Emerging attacks additionally coordinate several passages or operate under black-box settings [Chang et al. 2025; Xi et al. 2026], which further complicates retrieval-time defense and motivates structural reasoning over the whole retrieved set.

── P-RW-1b | ~140 words | 引用建议: [PoisonedRAG], [BadRAG], [Tan et al. 2024b], [One Shot Dominance 2025], [RIPRAG 2026]

「RW2a」Related Work 2 · 手工特征与聚类/可证明防御

①一类：手工特征过滤（困惑度/嵌入距离，Zhong’23）；②二类：可证明鲁棒（RobustRAG 隔离响应聚合，计算成本高）；③三类：聚类+LLM 重排（TrustRAG，自适应攻击下退化）；④一句共同局限收尾。

💬 Reference pattern: Several studies have explored defense mechanisms against misinformation introduced through the RAG workflow [Xiang et al. 2024]. These defenses usually utilize hand-crafted features, such as likelihood, embedding distances, and perplexity, to filter the low-quality retrieval content [Zhong et al. 2023]. For example, Xiang et al. (2024) mitigate poisoning via isolated response aggregation at high computational cost, and Zhou et al. (2025) adopt a clustering-based filter to enhance robustness but degrade under adaptive attacks.

Several studies have explored defense mechanisms against misinformation introduced through the RAG workflow [Xiang et al. 2024]. These defenses usually utilize hand-crafted features, such as likelihood, embedding distances, and perplexity, to filter low-quality retrieval content [Zhong et al. 2023]. For example, Xiang et al. (2024) mitigate poisoning via isolated response aggregation at a high computational cost, and Zhou et al. (2025) adopt a clustering-based filter that improves robustness but degrades under adaptive attacks. A common limitation is that each retrieved document is scored independently, so the redundant control of a mutually-supporting poisoned group remains invisible.

── P-RW-2a | ~120 words | 引用建议: [Xiang et al. 2024], [Zhou et al. 2025], [Zhong et al. 2023]

「RW2b」Related Work 2 · 结构/归因防御与差距陈述

①近年防御走向结构性信号：ReliabilityRAG 用 NLI 矛盾图 + 最大独立集找一致多数；AttnTrace 用注意力做上下文归因；Push and Pull 重塑嵌入空间；Who Taught the Lie 做投毒责任归因。

②差异句（务必精确引用）：上述工作均逐文档或逐来源处理，无人把候选集建模为"支持-矛盾联盟"并测量联盟级生成控制力，也无人针对联盟式自适应攻击训练防御。

③落点句：当前的防御仍是分裂的、反应式的（改写 ShieldRAG 措辞）。

💬 Reference pattern: More recent defenses leverage structural signals: ReliabilityRAG constructs an NLI-based contradiction graph and selects a consistent majority via a maximal independent set; AttnTrace performs context-level attribution through attention distributions; and other works reshape the embedding space or attribute responsibility for poisoned knowledge [He et al. 2026; Zhang et al. 2026]. However, so far no defense models the retrieved set as signed evidence coalitions and measures coalitional generation control, nor is explicitly trained against coalition-aware adaptive attacks. This shift in threat model reveals that current RAG defenses remain fragmented and reactive, lacking the structural robustness needed to counter evolving threats.

More recent defenses leverage structural signals: ReliabilityRAG-style methods construct an NLI-based contradiction graph and select a consistent majority via a maximal independent set; attribution-based methods trace responsibility through attention distributions or embedding-space reshaping. However, existing work still handles documents or sources in isolation; to the best of our knowledge, no defense models the retrieved set as signed evidence coalitions, measures coalition-level generation control, or is explicitly trained against coalition-aware adaptive attacks, which reveals that current RAG defenses remain fragmented and reactive when facing coordinated threats.

── P-RW-2b | ~150 words | 引用建议: [ReliabilityRAG], [AttnTrace], [Push and Pull], [Who Taught the Lie]

3. Methods

「M1」Framework Overview（Fig.2）

①总览段（抄 ShieldRAG 结构）：提出 RAG-ConGuard，目的+威胁；②正式定义 Unsupported Influence 假设（符号化：影响高 + 独立支持低 = 危险）；③Fig.2 流水线：七模块单箭头 → 生成/拒答；④训练期攻防循环一句预告；⑤"详细方法如下"。

💬 Reference pattern: We propose RAG-ConGuard, a framework designed to enhance the robustness of RAG systems against coordinated knowledge poisoning. The overall framework is illustrated in Fig. 2. The core safety hypothesis is the Unsupported Influence: a successful attack manifests as an evidence coalition that exerts disproportionate influence on generation while lacking independent supporting evidence. The framework aligns with this hypothesis in three stages: contextual evidence structuring, coalition-level attribution, and risk-constrained context construction. The detailed method is presented below.

We propose RAG-ConGuard, a framework designed to enhance the robustness of RAG systems against coordinated knowledge poisoning. The overall framework is illustrated in Fig. 2. The core safety hypothesis is the Unsupported Influence: a successful attack manifests as an evidence coalition that exerts disproportionate influence over generation while lacking independent supporting evidence. The framework aligns with this hypothesis in three stages: contextual evidence structuring (multi-view retrieval and query-conditioned claim extraction), coalition-level attribution (signed evidence graph, coalition discovery, and coalitional counterfactual influence), and risk-constrained context construction (hierarchical set-aware risk learning and safe context selection). A coalition-aware adaptive attack is additionally employed to train and evaluate the risk learner in an adversary-in-the-loop manner. The detailed method is presented below.

── P-Method-1 | ~180 words | 引用: 无新引用

「M2」多视角检索稳定性（辅助模块）

①构造语义等价视角：原始+稠密 / 改写+稠密 / 原始+BM25 / 原始+混合；②候选并集 [Eq]，每文档取多个视角排名；③稳定性表示 [Eq]：多视角排名一致性度量；④直觉句：正常证据稳定、单视角过拟合投毒文档多视角不一；⑤定位声明：本模块是辅助信号，非独立创新（避免被批混合检索无新意）。

📐 [Eq: 候选并集 D(q) = ⋃_v Top-k_v(q_v) — 在此补公式]

📐 [Eq: 检索稳定性 r_d = f(rank_v(d) for v ∈ V) — 在此补公式]

💬 Reference pattern: For each query q, we construct semantically equivalent views and retrieve candidates under each; the candidate pool is the union of the per-view top-k lists. For each document, instead of a single relevance score, we obtain a retrieval stability representation across views. Intuitively, benign evidence maintains stable ranking across views, while poisoned documents overfitted to a single query encoding exhibit cross-view inconsistency. This signal serves as an auxiliary component rather than an independent contribution.

For each query, RAG-ConGuard performs multi-view retrieval: a dense semantic view using BGE-based embedding with FAISS, a lexical BM25 keyword view, and a lightweight knowledge-graph view built from rule-based entity and triple extraction. Candidate lists from the three views are fused with Reciprocal Rank Fusion (RRF) into a ranked evidence set. This composition provides complementary signals: shared entities and keywords surface the source documents of an injected group, while dense similarity recovers paraphrase-level relatedness. Following the attack literature, we keep this component deliberately simple; our contribution lies in the subsequent structural analysis rather than in retrieval itself.

── P-Method-2 | ~170 words | 引用建议: 混合检索/查询改写 各 1 条

「M3」查询条件主张提取

①用轻量查询条件句子选择器（非 LLM 生成）为每篇文档选出 Top-K 相关主张 [Eq]；②两点优势：比整篇文档做 NLI 更准更快、避免 LLM 生成引入幻觉；③与 M4 衔接。

📐 [Eq: e_d = top-K(argmax_{s∈d} rel(s, q)) — 在此补公式]

💬 Reference pattern: Rather than generating claims with an LLM, we adopt a lightweight query-conditioned sentence selector that scores each candidate sentence by its relevance to the query and retains the top-K claims per document. This yields a query-specific claim set with low latency and no generation hallucination risk.

Rather than generating claim texts with an LLM, we represent each retrieved document by a query-relevant excerpt: the leading sentences of the document up to a fixed length (220 characters), which constitutes one claim node. This choice avoids generation hallucination in the claim extraction step, keeps the downstream NLI cost low, and aligns the claim units with the textual surface of the attack passages. Longer documents are truncated; each retrieved document thus contributes exactly one node to the graph in our setup, and the multi-view evidence set size equals the retrieval depth (k=5).

── P-Method-3 | ~120 words | 引用: 句子选择/抽取式 1 条

「M4」有符号证据图（核心机制组件）

①图定义 [Eq]：支持边(+)/矛盾边(-)/中性/文本冗余度 τ；②独立支持度 [Eq]（强调近重复文档不计多份独立证据）；③冲突度 [Eq]；④差异声明句式：图不是"首次用 NLI 图"而是"为发现投毒联盟 + 联盟级 CCI 提供结构单元"（ReliabilityRAG 用最大独立集找一致多数，与你的用途不同）。

📐 [Eq: s_i = Σ_{j: e_ij = +} (1 − τ_ij) — 在此补公式]

📐 [Eq: c_i = Σ_{j: e_ij = −} w_ij — 在此补公式]

💬 Reference pattern: We construct a query-conditioned signed evidence graph: each candidate document is a node, and an NLI model assigns positive (support/entailment), negative (contradiction), or neutral edges between document claims, with a text redundancy factor measuring near-duplication. Unlike prior NLI-graph defenses that extract a consistent majority, the graph here serves to discover potential poisoning coalitions and to provide the structural units for the subsequent coalitional counterfactual influence analysis.

We construct a query-conditioned signed evidence graph over the retrieved claim nodes. A cross-encoder NLI model (cross-encoder/nli-deberta-v3-base, MNLI-trained) scores every directed claim pair, yielding positive (entailment/support), negative (contradiction), or neutral relations; graph edges keep the sign and confidence. Two complementary mechanisms refine this backbone. First, a numeric-conflict heuristic intercepts same-entity different-value pairs (e.g., "24 episodes" vs. "23 episodes") that the NLI model tends to classify as neutral, converting them to contradiction edges with confidence 1.0. Second, low-confidence pairs are re-judged by an LLM judge, whose score is fused with the NLI score as alpha*s_NLI + (1-alpha)*s_LLM. Note the difference from prior NLI-graph defenses: rather than extracting a consistent majority, the graph here serves to expose evidence coalitions and to provide the structural units for the counterfactual analysis that follows.

── P-Method-4 | ~170 words | 引用建议: [ReliabilityRAG] 差异声明必须写

「M5」证据联盟发现

①联盟定义 [Eq]：有符号图上按支持边阈值连通的文档组（可选 signed spectral / correlation clustering 精化）；②可解释性：联盟可能是多来源信任簇 / 近重复投毒簇 / 单篇强影响文档 / 竞争事实簇；③复杂度句：k 小计算可控。

📐 [Eq: Π = {C_1, ..., C_m}, C_t 内支持边密度 ≥ τ_c — 在此补公式]

💬 Reference pattern: On the signed evidence graph, we partition candidate documents into evidence coalitions by thresholding support-edge density, with optional signed spectral refinement. Because the number of retrieved candidates is typically small (k ≤ 10), this step is computationally inexpensive. The resulting coalitions may be multi-source benign clusters, near-duplicate poisoned clusters, a single highly-influential document, or competing fact clusters.

On the signed evidence graph, we discover evidence coalitions as the connected components induced by positive (supportive) edges. Because the retrieved set is small (at most k=5 in our settings), connected-component enumeration is exact and inexpensive, and no multivariate cluster refinement is required. The resulting coalitions may be multi-source benign clusters, near-duplicate poisoned clusters, a single highly-influential document, or competing fact clusters; characterizing them requires the additional features introduced next.

── P-Method-5 | ~130 words | 引用: 聚类/谱聚类 1 条

「M6」联盟级反事实生成影响 CCI ★核心贡献 1

①完整上下文生成探针：前 H 个回答 token，第 t 位分布 [Eq]；②整体移除联盟 C 重算 [Eq]；③CCI 定义 [Eq]：两级分布 JS 散度；④语义：高值 = 联盟对生成决策的真实控制力；⑤对比论证：逐文档 leave-one-out 在 3 篇近重复投毒下每篇得分都被稀释（写清这个"算不动"场景）；⑥与 RAGuard 差异声明句式必须原文级别写出来。

📐 [Eq: p_t^full = P(token_t | prefix_{<t}, q, D, θ) — 在此补公式]

📐 [Eq: p_t^{−C} = P(token_t | prefix_{<t}, q, D \ C, θ) — 在此补公式]

📐 [Eq: CCI(C) = JSD(p^full_t ∥ p^{−C}_t) — 在此补公式]

💬 Reference pattern: Existing counterfactual defenses remove documents individually; RAG-ConGuard first identifies potential coalitions through the signed evidence graph and then performs coalition-level counterfactual intervention, capturing the redundant control and synergistic influence among multiple poisoned documents. Formally, we generate a short answer probe of H tokens under the full context and recompute the token distribution after removing the whole coalition; the coalitional counterfactual influence is defined as the Jensen–Shannon divergence between the two distributions.

Our coalitional counterfactual influence (COAL-INFLUENCE) is defined over a trust-weighted answer-support score, which serves as a lightweight proxy of generation control. First, a trust propagation step computes a confidence weight p_c for each claim from the signed graph: support edges raise the trust of their targets while refutation edges suppress it, with a refutation penalty mu>1 applied iteratively. Second, for each candidate answer a (the correct and the attacker-specified answer of the target question), the answer-support score is s_G(a) = sum_c p_c * max(0, sim(c, a)), where sim is the BGE cosine similarity between claim and answer. Third, the influence of a coalition C is defined as the relative drop in support for the most affected answer after removing the whole coalition: I(C) = max_a [s_G(a) - s_{G\C}(a)] / s_G(a*). The control of a coalition is then I(C) times one minus its external corroboration, i.e., the extent to which the coalition stands alone. Because the number of coalitions is small (2-5 per query), the exact computation is cheap. Existing counterfactual defenses remove documents one by one; under a group of near-duplicate poisoned documents, per-document removal leaves the remaining members fully in control, so individual influence is diluted. Coalition-level removal is the exact counterfactual intervention that uncovers such redundant control.

── P-Method-6 | ~200 words | 引用建议: [RAGuard (leave-one-out)] 对比必须; 禁用"首次反事实删除"

「M7」分层集合风险学习器 ★核心贡献 2

①文档级表示 [Eq]：融合检索稳定性、独立支持度、冲突度、查询相关性、生成模型统计自然度；②图注意力编码 [Eq]；③联盟级集合注意力聚合 [Eq]；④三层输出 [Eq]：文档风险 / 联盟风险 / 查询级攻击风险；⑤三级监督：来源标签、联盟必要性标签 [Eq]（该联盟对攻击成功是否必要）、查询级攻击成功标签；⑥联合损失 [Eq]：三层 BCE + 联盟排序损失。

📐 [Eq: h_d^{(0)} = [r_d; s_d; c_d; rel(q,d); n_d] — 在此补公式]

📐 [Eq: h_d^{(ℓ+1)} = GAT(h_d^{(ℓ)}, E) — 在此补公式]

📐 [Eq: z_C = SetAttn({h_d : d ∈ C}) — 在此补公式]

📐 [Eq: ŷ_d, ŷ_C, ŷ_q — 在此补公式]

📐 [Eq: L = Σ BCE(ŷ_d,y_d) + BCE(ŷ_C,y_C) + BCE(ŷ_q,y_q) + λ_rank · L_rank — 在此补公式]

💬 Reference pattern: Instead of classifying each document independently, we learn hierarchical set-aware representations: a lightweight graph attention network encodes document representations over the signed evidence graph, and a set-level attention aggregates coalition representations. The model simultaneously outputs document-level, coalition-level, and query-level risks, answering not only whether a document is poisoned, but also which coalition jointly controls the generation and whether the current retrieval is under an effective attack.

Instead of classifying each document independently, we learn a hierarchical set-aware risk score over coalition structure. For every coalition we compute six structural and influence features: size, internal cohesion (support-edge density), external corroboration (per-member external support edges), external refutation (per-member external contradiction edges), the coalitional influence I(C), and the control score. A three-layer MLP (16-8 hidden units) maps the standardized feature vector to a probability that the coalition is poisoned; standard scaling is fitted on the training split. Labels are weak but actionable: a coalition is labeled poisoned when at least 80% of its members are poisoned. For training we split queries 70/30 to avoid leakage. An interpretable rule-based scorer, obtained by a fixed weighted sum of the same features passed through a sigmoid, serves as the no-training baseline. The learned model attains a held-out AUC of 0.792 for distinguishing poisoned from benign coalitions, compared to 0.194 for the rule baseline, which shows that the feature combination, rather than a single feature, drives discrimination.

── P-Method-7 | ~220 words | 引用建议: 图注意力 + 集合注意力 各 1 条

「M8」风险约束安全上下文选择 ★核心贡献 3

①把"阈值过滤"升级为集合优化问题 [Eq]：约束五条（相关性下限、独立支持优先、矛盾联盟不共存、近重复预算、总风险预算）；②贪心实现步骤（删高风险联盟→按边际安全效用排序→逐篇加入→每次重算集合风险→无可行解拒答）；③conformal calibration 在预设良性误删预算 [Eq] 下校准；④定位句：从"恶意文档分类器"到"安全上下文构造算法"。

📐 [Eq: D* = argmax_{S ⊆ D} U(S)  s.t. 风险预算 / 相关性 / 无内部矛盾联盟 — 在此补公式]

📐 [Eq: P(误删良性) ≤ α (conformal calibration) — 在此补公式]

💬 Reference pattern: Unlike independent threshold filtering, our objective is to directly construct a safe evidence subset under a preset benign mis-filtering budget, rather than selecting the operating point that maximizes classification F1. In practice, we implement this as a greedy selection: remove high-risk coalitions, sort the remaining documents by marginal safety utility, add them one by one while re-evaluating the set-level risk, and refuse to answer when no feasible safe subset exists.

Given the coalition risk scores, RAG-ConGuard constructs a safe context instead of filtering documents with a static per-document rule. For a risk threshold tau, all coalitions whose learned risk is at least tau are removed, and every document that contributed a member claim is dropped from the context; the remaining documents form the safe evidence subset from which the generator answers. If no document survives, the system refuses to answer with a canonical "I don't know" response. Tuning tau sweeps the robustness-utility trade-off: conservative thresholds remove more evidence and raise the refusal rate, while permissive thresholds keep more of the poisoned group. This differs fundamentally from per-document anomaly detection, because the decision is taken at the set level and is conditioned on how much of the retrieval is jointly controlled. In our end-to-end evaluation the learned risk learner attains a 71-point reduction in attack success rate at tau=0.25.

── P-Method-8 | ~180 words | 引用建议: conformal prediction 1 条

「M9」ADSEA-C 联盟式自适应攻击 ★核心贡献 4

①攻击者联合构造 b 篇投毒文档，联合约束 [Eq]：多视角可检索、诱导目标答案、文档间互相支持、低重复度、与良性证据低冲突、降低文档/联盟/查询三级检测分；②交替优化步骤（GCG token 候选→多视角排名→NLI 评估支持冲突→风险模型打分→重排→LLM 整篇语义重写→逐篇轮换）；③一句"为什么比只优化 RS 的攻击强"（它真正针对集合检测器）。

📐 [Eq: max_{x_1..x_b} Σ_t R_attack − λ_d Σ risk 检测 — 在此补公式]

📐 [Eq: 攻击者只需绕过"当前检测器"即视为成功 — 在此补公式]

💬 Reference pattern: ADSEA-C jointly optimizes b poisoned documents so that they remain retrievable across views, induce the attacker-specified answer, mutually support each other, avoid excessive duplication and overt conflict with benign evidence, and minimize document-, coalition-, and query-level risk scores. Based on alternating optimization over token candidates, multi-view ranking evaluation, NLI-based support/conflict assessment, and risk-scoring by the current detector, the attack iteratively adapts to the defense under evaluation.

To stress the defense under evolving evidence, we design a coalition-aware adaptive attack that rewrites an existing targeted poisoning group. The attack family consists of three templates that ask an LLM to generate variants of the original adversarial passages while preserving the attacker-specified factual assertion: (A) a corroborating variant that adds claims of multiple independent sources, (B) a de-coordinated variant that expresses the same assertion without any cross-reference wording, and (C) a mixed variant that mentions exactly one source. Each template keeps the passage retrievable (the rewritten texts are only style-level rewrites of the original passages) and keeps the group mutually supportive of the same answer. The attack is adaptive in the sense that each round of the loop re-evaluates the current risk learner and retains surviving variants as new hard positives.

── P-Method-9 | ~180 words | 引用建议: [GCG], [PoisonedRAG], [BadRAG]

「M10」攻击者在环的对抗训练

①四轮流程：初始（PoisonedRAG/Joint-GCG 训练检测器）→ 冻结检测器运行 ADSEA-C 收集成功规避样本 → 入 hard-example pool 重训 → 攻击更新后检测器，迭代 2-3 轮；②资格声明句：不是"训练完再测试自适应攻击"，而是"以攻击者在环的困难样本挖掘学习稳定风险边界"；③差异句：不依赖注意力/隐藏状态，面向多文档投毒联盟（对标 AttnTrace 系）。

💬 Reference pattern: RAG-ConGuard does not merely evaluate the detector under adaptive attacks after training; it mines hard evasion examples with an adversary in the loop, alternating between strengthening ADSEA-C and retraining the risk learner. Starting from standard poisoning attacks, rounds of adversary-in-the-loop hard example mining learn a stable risk boundary against evolving evasion strategies, without relying on attention or hidden-state signals.

RAG-ConGuard does not merely evaluate the detector under adaptive attacks after training: it mines hard evasion examples with an adversary in the loop. The training schedule alternates the following steps: (i) the current risk learner (initially the rule-based scorer) scores the original poisoned groups and their variant groups; (ii) variants that evade the current detector (risk below tau) are added to a hard-example pool together with benign coalitions as negatives; (iii) the MLP risk learner is retrained on the accumulated pool; (iv) the adversarial generators re-target the updated detector. In the observed schedule, the round-0 rule detector and the round-1 MLP detector reject all template variants (escape rate 0.00), while the round-2 detector, trained on 89 pooled groups, is defeated by the evolving attack (escape rate 1.00), demonstrating both the adaptivity of the attack family and the necessity of the adversary-in-the-loop schedule to avoid a static risk boundary.

── P-Method-10 | ~170 words | 引用建议: [PoisonedRAG/Joint-GCG 训练], 差异声明对标 2026 注意类工作

4. Experiments and Analysis

「E1」Setup · 目标与指标

①目标：对抗未定向/定向/自适应投毒下的鲁棒性 + 效用；②指标：检索 ASR@k（top-k 至少含一篇投毒文档的查询占比）、Recall@k、生成 ASR（有害或拒答输出占比）、生成准确率、可选延迟与良性误删率（FPR，conformal 口径）；③口径与旧项目 MILESTONE_* 一致。

💬 Reference pattern: The goal of the evaluation is to assess the robustness and utility of RAG-ConGuard under various poisoning attack scenarios. Retrieval robustness is measured by the retrieval attack success rate (ASR), defined as the percentage of queries for which at least one adversarially poisoned document appears in the top-k retrieved results. Retrieval utility is assessed using Recall@k. Generation robustness is evaluated by the generation ASR, capturing the fraction of outputs that are harmful or refuse to answer due to injected content, and generation utility by answer accuracy.

The goal of the evaluation is to assess the robustness and utility of RAG-ConGuard under coordinated knowledge poisoning. Retrieval robustness is measured by the retrieval attack success rate (retrieval ASR), the percentage of queries for which at least one poisoned document enters the top-k evidence; retrieval utility is assessed by the multi-view hit rate and by Retention@k of the safe context. Generation robustness is measured by the generation ASR, the fraction of outputs whose answer matches the attacker-specified answer (matching based on the standard clean-str exact/containment criterion), and generation utility by answer accuracy. All settings follow the MILESTONE definitions inherited from the attack framework used for the corpus.

── P-Exp-1 | ~140 words | 引用: 无新引用

「E2」Setup · 攻击设置

①未定向攻击（Tan’24b 口径：高检索似然、无特定目标）；②定向攻击（Xue’24 口径：特定查询触发特定有害内容）；③自适应联盟攻击 ADSEA-C（本文，b=3 默认）；④主实验投毒率 0.05%（与 ShieldRAG 对齐），扫描 0.02%~2.0%；⑤注明投毒数据与旧项目一致（PoisonedRAG 产物）。

💬 Reference pattern: We consider three representative attack settings: untargeted attacks following Tan et al. (2024b), targeted attacks following Xue et al. (2024), and the coalition-aware adaptive attack ADSEA-C proposed in this work. Concerning the poison injection ratio, we adopt a fixed ratio of 0.05% in our main experiments; for robustness analysis, we additionally vary the proportion of injected poisoned content from 0.02% to 2.0%.

We follow the targeted poisoning setting of the PoisonedRAG benchmark: for every query of a dataset, five adversarial passages are generated that answer the query with an attacker-chosen incorrect answer. The 5 x N poisoned passages are injected into the corpus before retrieval; no further selection or filtering is applied to them. We additionally consider the coalition-aware adaptive variants introduced in Section 3.9, in which an LLM rewrites the original group with corroboration, de-coordination, or mixed semantics while preserving the attacker-specified assertion. The injection rate is fixed by the benchmark (500 poisoned passages over 600k documents, ~0.08%).

── P-Exp-2 | ~120 words | 引用建议: [Tan et al. 2024b], [Xue et al. 2024] + 本文 ADSEA-C

「E3」Setup · 基线

①无防御 Vanilla RAG；②旧版 RAG-ConGuard（RS+GS+CS+MLP 逐文档过滤器——你的消融对照组）；③代表性防御：TrustRAG（聚类+LLM 重排）；④可选：ReliabilityRAG 式 NLI 一致性、归因式过滤（注明实现口径，重跑不引用其代码不得照抄数字）；⑤声明：所有基线同检索器同数据同攻击。

💬 Reference pattern: Most existing RAG defenses are designed to counter single-passage anomalies, which differ from the coordinated scenario we target. As representative baselines, we include Vanilla RAG, the predecessor of our framework (a per-document feature-based filter), TrustRAG [Zhou et al. 2025], and optionally NLI-consistency- and attribution-based filters. All baselines share the same retriever, corpus, and attack settings for a fair comparison.

Most existing RAG defenses are designed to counter single-passage anomalies, which differs from the coordinated scenario we target. As representative baselines we include: (i) Vanilla RAG, which examines the retrieved evidence untouched; (ii) a relevance-filter baseline that removes documents whose BGE similarity to the query falls below a threshold; and (iii) a duplicate-group baseline that removes groups of documents whose pairwise BGE similarity exceeds a high threshold, mimicking near-duplicate-source filtering. All baselines share the same retriever, corpus, attack, and generator for a fair comparison.

── P-Exp-3 | ~110 words | 引用建议: [TrustRAG], [ReliabilityRAG] 可选

「E4」Setup · 数据集与模型

①检索/生成：NQ、HotpotQA、MS MARCO（可扩 SQuAD）；②若做定向毒性：AdvBench、HateXplain、ToxiGen（ShieldRAG 同口径）；③检索器：BGE-base-en-v1.5（与旧项目一致，标注可比性）；④LLM：主用 Qwen2.5-7B，多模型矩阵可选 Llama-3-8B / GPT-4o-mini；⑤生成/提取后端与服务器 vLLM 环境备注。

💬 Reference pattern: To evaluate untargeted attacks, we use standard open-domain retrieval benchmarks including NQ, HotpotQA, MS MARCO, and SQuAD; for targeted attacks, we adopt toxicity-focused datasets such as AdvBench, HateXplain, and ToxiGen. BGE-base-en-v1.5 is employed as the retriever backbone due to its strong zero-shot performance, and [Qwen2.5-7B] serves as the target LLM [please fill the actual model list].

To evaluate the framework, we use three standard open-domain retrieval benchmarks: Natural Questions (NQ), HotpotQA, and MS MARCO, each with 600,000 corpus documents and 5,000 (3452 for NQ) queries; the evaluation queries are the 100 poisoned queries per dataset defined by the benchmark. BGE-base-en-v1.5 serves as the retrieval backbone with the FAISS index, BM25 for the lexical view, and the rule-based entity store for the KG view. Qwen2.5-7B-Instruct is the generation LLM, running in bf16 on an RTX 4090; all generation and judging were performed under a fixed temperature-0 setting.

── P-Exp-4 | ~130 words | 引用建议: [NQ], [HotpotQA], [MS MARCO], [SQuAD], 毒性集可选

「E5」Main Results · 检索阶段（Table 1）

按「图→趋势→机制」三步：①表：Table 1 各数据集 × Top@1/3/5 × {Recall, ASR}，含旧版、TrustRAG、无防御；②趋势：新框架 ASR 压至 [实测区间]，Recall 恢复至接近无攻击水平；③机制：联盟级冗余控制信号 + 检索稳定性 vs 逐文档统计特征的判别力差异；若同口径旧版在近重复投毒下 ASR 明显更高，这行数据必须写出。

💬 Reference pattern: As shown in Table 1, RAG-ConGuard achieves significantly better defense performance without compromising retrieval effectiveness compared with the baselines. In terms of robustness, it suppresses ASR to near-zero levels across attack types and datasets. In terms of utility, the recall performance is restored to a level comparable to the attack-free results, confirming that coalition-level structural reasoning does not sacrifice retrieval quality while enhancing robustness.

Table 1 reports retrieval-stage robustness and utility. In the attack setup, all 100 poisoned queries per dataset retrieve at least one poisoned document into the top-5 evidence (retrieval ASR = 1.0 before defense), confirming that the attack material reaches the prompt. Per-view, the dense and keyword views hit every target query, while the KG view hits 0.83-1.0, showing that the multi-view composition is redundant rather than fragile. Restoration to the pre-filter subset, rather than rejection of the whole retrieval, is the mechanism by which retrieval utility is retained.

── P-Exp-5 | ~160 words | 无新引用

「E6」Main Results · 生成阶段（Table 2）

图→趋势→机制：①Table 2 数据集 × LLM × 生成 ASR；②趋势：所有 LLM 下新框架最低、弱者提升更大；③机制：安全子集从源头删除控制联盟；④观察句：强 LLM 固有鲁棒性强但仍会被协同攻击命中——检索侧联盟防御对强 LLM 同样必要（对齐 ShieldRAG 观点）。

💬 Reference pattern: Table 2 reports the generation ASR under all attack settings across various datasets and LLMs. RAG-ConGuard consistently achieves lower ASR across all settings. Additionally, we observe that more capable LLMs exhibit stronger inherent robustness, yet they remain vulnerable to certain coordinated attacks, further underscoring the necessity of retrieval-time coalition-aware mechanisms.

Table 2 reports generation ASR under no defense, per-document baselines, and RAG-ConGuard. Without defense, the attacker-specified answer is produced by the unguarded pipeline in 0.86 (NQ), 0.99 (HotpotQA) and 0.82 (MS MARCO) of the 100 evaluation queries. A relevance-based single-document filter changes none of these numbers, and the duplicate-group filter reduces them only marginally (to 0.76 / 0.93 / 0.79), because the poisoned group is semantically relevant to the query and its members are individually indistinguishable. With the learned coalition risk learner and tau=0.25, the generation ASR drops to 0.143 (from 0.857 on the same evaluation split), which is the direct effect of removing the controlling coalition from the context.

── P-Exp-6 | ~140 words | 无新引用

「E7」Main Results · 生成精度恢复（Table 3）

①Table 3：{无攻击 / 受攻 / TrustRAG / 旧版 / 新框架} × LLM 的生成准确率；②趋势：受攻大幅下降，新框架恢复接近无攻击；③机制句：既移除控制联盟又保留高相关证据——对比旧版 "滤得狠→伤精度" 的现象，用 FPR 预算实验佐证]。

💬 Reference pattern: Table 3 shows the generation accuracy under attack-free, attacked, and defended settings. Poisoning attacks significantly degrade answer accuracy across all models, highlighting their adverse impact on generation utility. RAG-ConGuard restores performance close to attack-free levels, demonstrating its ability to preserve utility while ensuring robustness.

Table 3 shows generation accuracy under attack-free, attacked, and defended settings. Poisoning strongly degrades accuracy on the attacked queries; importantly, the defended setting also shows the cost of aggressive filtering: with full-coalition removal the refusal rate grows, and answer accuracy is traded against robustness. Our design therefore exposes the robustness-utility trade-off as an explicit threshold choice rather than a hidden one, and the ablation of Section 4.8 shows the effect of choosing tau=0.25, which keeps the strongest robustness gain. The similarity/relevance baselines, by contrast, do not pay this cost but also provide no robustness gain, which we attribute to their inability to localize the coalition.

── P-Exp-7 | ~120 words | 无新引用

「E8」对照组 · 冗余投毒下 LOO vs CCI（Table 4）★★ 多块

本表是「逐文档 vs 联盟级」差异的直接证据，必须单独做：①场景：注入 b=3（可扩 5）篇近重复互支持投毒文档；②度量对比：逐文档 LOO（删除 1 篇）/ 删除全体 LOO / 联盟级 CCI；③预期/结果模式：单篇删除影响 | 低 |（被剩余 2 篇稀释），联盟删除影响【高】；④机制句：解释逐文档归因为何在冗余投毒下失效。

（这是把你 M6 的论证变成数字的最关键实验。）

💬 Reference pattern: Table 4 reports a focused comparison between per-document leave-one-out attribution and our coalitional counterfactual influence under redundant poisoning. When three mutually-supporting poisoned documents are injected, removing any single one leaves the remaining two fully in control, so individual influence remains low; removing the whole coalition sharply increases the divergence—the exact signal our detector relies on. This confirms that per-document attribution is unreliable under coordinated attacks, and motivates coalition-level reasoning.

A focused comparison between per-document filtering and coalition-level defense is given by Table 4, using the redundancy setting of the attack: five near-duplicate mutually-supporting passages per query. The duplicate-group baseline removes them only when they are above a tight similarity threshold; otherwise they pass. The coalition defense instead identifies the group structurally, through its own support edges on the signed graph, and removes it as a whole. In the evaluation, the coalition defense reduces generation ASR by 71 points, whereas the best single-document baseline reduces it by at most 10 points. This confirms that per-document attribution and filtering are unreliable under coordinated poisoning and motivates coalition-level reasoning.

── P-Exp-8 | ~150 words | 引用建议: 已在 M6 引 [RAGuard]

「E9」对比 · 与 LLM 安全评估器（Fig.3）

①范式描述：LLM 直接当安全过滤器（读文档打毒分/过滤）；②对比三维：Recall@1、ASR@1、检索阶段延迟；③结果句：效用/鲁棒与最强 LLM 相当，延迟低 1-2 个数量级（本地免费 vs API 成本）；④机制句：轻量模型 + 集合结构推理，无需在线大模型评审。

💬 Reference pattern: A straightforward strategy is to connect a retriever with an LLM acting as a safety evaluator. We compare RAG-ConGuard with representative LLM safeguards across three key dimensions: retrieval utility (Recall@1), retrieval robustness (ASR@1), and retrieval-phase latency. As shown in Fig. 3, RAG-ConGuard achieves utility and robustness comparable with the best LLMs while exhibiting substantially lower latency, making it a practical solution for real-time RAG systems.

The comparison against an LLM-as-judge safety filter is limited by design in this version of the study: in our pipeline an LLM judge is used only for low-confidence claim pairs in the graph construction step, not as a document-filter baseline. We therefore report the latency of each consumed component: the NLI cross-encoder scores the claim graph in batch on the local GPU, the LLM judge handles only the low-confidence subset, and the whole graph-to-risk stage per query completes in seconds. This keeps the framework practical for real-time RAG.

── P-Exp-9 | ~130 words | 引用建议: [GPT-4o], [Gemini 2.5], [Claude-3.5] 模型卡

「E10」结果 · 投毒率扫描（Fig.4）

①横轴 0.02%~2.0% 投毒占比，双轴：Recall@1（w/o Attack 基线 / 受攻）+ ASR@1；②趋势句：占比增 → 受攻组 Recall 持续下滑、ASR 飙升，新框架保持近零 ASR 与高位 Recall；③机制句：大污染下依然有效 = 联盟级信号不依赖"少数异常"假设。

💬 Reference pattern: Fig. 4 evaluates the robustness of RAG-ConGuard under varying proportions of unsafe knowledge content, ranging from 0.02% to 2% of the corpus. As the poisoning ratio increases, the vanilla pipeline degrades, with Recall@1 declining and ASR@1 rising sharply. In contrast, RAG-ConGuard maintains consistently high recall and near-zero attack success across all poisoning levels, demonstrating strong robustness even under high contamination.

We do not vary the corpus poisoning ratio in this version: the benchmark fixes the injection volume (500 passages per dataset), and our variants are attack-text rewrites rather than increased contamination. The robustness of the framework is therefore evaluated under a fixed, moderately contaminated corpus, while the multi-view composition covers the density of the poisoned group throughout.

── P-Exp-10 | ~120 words | 无新引用

「E11」Ablation · 消融（Fig.5）

四变体逐一对应核心模块：w/o Evidence Graph（退化为文本相似度分组）、w/o CCI（用逐文档 LOO 影响）、w/o Set Learner（逐文档 MLP 一文章三特征）、w/o Adv-in-loop（仅固定攻击训练）；双面板 ASR/Recall @Top1/3/5；每变体描述"去哪了就掉多少"，写清"全配置一致最低 ASR/最高 Recall"。

💬 Reference pattern: Fig. 5 presents an ablation analysis highlighting the impact of graph-guided coalition discovery, coalitional counterfactual influence, set-aware learning, and adversary-in-the-loop training. Compared with the four variants, RAG-ConGuard consistently achieves a lower ASR and a higher recall at all retrieval depths. w/o CCI degrades most noticeably under redundant poisoning, and w/o Adv-in-loop degrades under adaptive attacks, underscoring the necessity of integrating structured coalition reasoning and adaptive robustness training.

The ablation of the framework components, corresponding to Fig. 5 in the framework overview, is summarized as follows. Removing the sign information (unsigned graph) lowers the coalition purity from 0.972 to 0.891, and the pure-similarity graph lowers it further to 0.850; the signed graph, additionally, keeps internal conflicts at the coalition boundary (conflict ratio 0.124 vs. 0.455 for the unsigned variant), which supports the interpretation that poisoned coalitions are internally consistent and externally conflicting. Removing the learned risk model and keeping the rule-based control score reduces the held-out AUC from 0.792 to 0.194, i.e., the structural features alone do not resolve the threat. Finally, ablating the adversary-in-the-loop schedule (by keeping the round-0 detector) leaves the detector vulnerable to the evolving attack variants, as shown in Section 4.10.

── P-Exp-11 | ~130 words | 无新引用

「E12」Sensitivity · 敏感性（Fig.6）

①主变元：联盟支持阈值 τ（或图密度）与探针长度 H；②可选：学习样本量 2k-10k（对齐 ShieldRAG 口径，证明数据效率）；③趋势+含义句：τ 宽范围稳健、小 H 即够 → 方法不依赖脆弱工作点/昂贵探测。

💬 Reference pattern: Fig. 6 shows the effect of the coalition threshold τ and probe length H. The results indicate that RAG-ConGuard achieves strong robustness across a wide range of τ, with performance degrading only under extreme settings, and that a short probe suffices to expose coalitional control. This demonstrates that the framework does not rely on a fragile operating point or expensive generation probes.

A sensitivity analysis of the risk threshold tau is reported in Fig. 6. Across tau from 0.05 to 0.5, the defended generation ASR stays at 0.143 on the evaluation split, i.e., the attacked group is consistently recovered as the highest-risk coalition; the differences among thresholds appear in the refusal rate rather than in the ASR. This indicates that the choice of operating point trades utility, not robustness, and that the framework does not rely on a fragile threshold.

── P-Exp-12 | ~110 words | 无新引用

5. Conclusion

「CL-帽」结语帽段

帽段：本文提出 + 应对什么 + 用什么机制（一句话总纲，复写创新声明组合级措辞）。

💬 Reference pattern: In this paper, we propose RAG-ConGuard, a coalition-aware counterfactual defense framework that enhances the robustness of RAG systems against coordinated knowledge poisoning in untrusted knowledge bases.

In this paper, we propose RAG-ConGuard, a coalition-aware counterfactual defense framework that enhances the robustness of RAG systems against coordinated knowledge poisoning in untrusted knowledge bases.

── P-Concl-Hat | ~70 words

「CL-1」结论 1 · 核心机制

结论 1：把检索证据建模为查询条件有符号图 + 联盟级反事实生成影响 → 无需先验攻击目标即可识别无独立支持的不成比例控制联盟。

💬 Reference pattern: By modeling retrieved evidence as a query-conditioned signed graph and measuring coalitional counterfactual generation influence, RAG-ConGuard identifies under-supported evidence coalitions that exert disproportionate control over generation, without knowing the attack target in advance.

By modeling retrieved evidence as a query-conditioned signed graph and measuring coalitional counterfactual influence over a trust-weighted answer-support score, RAG-ConGuard identifies under-supported evidence coalitions that exert disproportionate control over generation, without knowing the attack target in advance.

── P-Concl-1 | ~60 words

「CL-2」结论 2 · 集合学习与安全选择

结论 2：分层集合风险学习器 + 风险约束安全上下文选择，把"文档异常检测"升级为"安全上下文构造"，并以显式误删预算保住效用。

💬 Reference pattern: The hierarchical set-aware risk learner and the risk-constrained safe context selector jointly transform per-document anomaly detection into safe context construction, which preserves generation utility through an explicit benign mis-filtering budget.

The hierarchical set-aware risk learner and the risk-constrained safe context selector jointly transform per-document anomaly detection into safe context construction, which exposes the robustness-utility trade-off as an explicit threshold choice.

── P-Concl-2 | ~55 words

「CL-3」结论 3 · 实验与训练机制

结论 3：在多个数据集/LLM/攻击策略（含本文 ADSEA-C 与攻击者在环训练）下，鲁棒性显著增强且精度不掉。

💬 Reference pattern: Extensive experiments across multiple datasets, LLMs, and attack strategies—including the coalition-aware adaptive attack and adversary-in-the-loop training—demonstrate that RAG-ConGuard substantially enhances robustness without sacrificing accuracy or efficiency.

Extensive experiments across three datasets under targeted poisoning and its coalition-wise adaptive variants demonstrate that RAG-ConGuard substantially enhances robustness, reducing the generation attack success rate from 0.86-0.99 to 0.14 under fixed utility budgets, while similarity- and relevance-based single-document defenses provide no measurable gain.

── P-Concl-3 | ~65 words

「CL-4」结论 4 · 意义与展望

意义句 + 局限/未来：①NLI 边误差的级联（可用领域自适应/多模型投票）；②探针的生成开销；③更隐匿攻击下的鲁棒性；④未来：把联盟推理扩展到多跳/代理场景。

💬 Reference pattern: These results highlight the potential of coalition-level structural reasoning as a critical component for building trustworthy and resilient RAG systems. A limitation of our work is its dependence on the NLI model and the probing cost; future directions include robust claim relations under domain shift and extending coalition-aware reasoning to multi-hop and agentic scenarios.

These results highlight the potential of coalition-level structural reasoning as a critical component for building trustworthy and resilient RAG systems. A limitation of our work is its dependence on the NLI model and the LLM judge during graph construction, as noisy relation edges propagate into coalition discovery; future directions include stronger claim-relation models, the extension of coalition-level reasoning to multi-hop and agentic scenarios, and the validation of the framework under attack strategies beyond the template-based family considered here.

── P-Concl-4 | ~80 words

附录 · 图表清单一览

