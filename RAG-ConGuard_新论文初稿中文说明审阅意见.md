# RAG-ConGuard 新论文初稿中文说明审阅意见

我已经通读了你上传的《RAG_论文模仿框架.docx》，重点只看了其中中文说明部分（也就是你的写作规划、方法设计说明、实验规划，而不是英文 Reference pattern）。整体评价：**这个框架的研究思路已经明显超过之前的 RAG-ConGuard++，方向更接近 CCF-B / SCI 一区论文的写法，但目前中文设计中存在几个“审稿人会攻击”的地方，需要提前修正。**

下面我按照“合理性、创新性、风险点”逐部分评价。

---

# 总体评价

先说结论：

## 研究定位

目前框架：

> RAG-ConGuard: Coalition-Aware Counterfactual Defense against Coordinated Knowledge Poisoning in RAG

核心思想：

> 投毒不是单篇异常，而是多个文档形成联盟，在缺少独立支持情况下共同控制生成。

这个问题定义是合理的，而且比旧版：

> RS + GS + CS + MLP 文档分类

高一个层次。

因为旧版本质：

```text
document-level detection
```

现在升级：

```text
evidence-set-level reasoning
```

这是一个真正的方法升级。

文件中提出的核心假设：

> Unsupported Influence：缺乏独立支持的异常生成影响

这个概念是目前全文最重要的东西。

我认为：

**这个方向可以支撑 CCF-B / SCI二区甚至冲一区尝试。**

但是：

目前设计里有几个地方需要收紧，否则容易被审稿人认为：

> 概念很好，但实现不可验证。

---

# 一、标题和核心概念

## 合理之处

三个标题中，T1 最好：

> RAG-ConGuard: Coalition-Aware Counterfactual Defense against Coordinated Knowledge Poisoning in Retrieval-Augmented Generation

原因是它符合顶会常见的命名方式：

```text
系统名 + 方法特点 + 攻击问题
```

例如 ShieldRAG 使用了：

> Safeguarding Retrieval-Augmented Generation...

你的标题使用：

> Coalition-Aware Counterfactual Defense

方法特点表达得很明确。

## “Coalition”一词需要谨慎

目前全文大量使用：

- coalition
- coordinated poisoning
- coalition-level influence

审稿人可能会问：

> Why is this a coalition rather than simply correlated documents?

因为 coalition 在机器学习和博弈论语境中具有较强的理论含义。

你必须明确：这里的联盟不是简单的“相互支持文档集合”，而应该定义为：

> a set of retrieved documents whose joint removal causes disproportionate generation change while individual removal shows limited effect.

也就是说，你的 coalition 定义不能只是“相互支持”。

否则，ReliabilityRAG 一类基于矛盾图或一致性图的方法可能会直接质疑：

> 你只是给文档聚类换了一个名字。

你的文件中 M5 暂时将联盟描述为：

> 按支持边阈值连通的文档组。

这里风险较大。

不建议定义为：

```text
coalition = connected component
```

更合理的定义应包含：

```text
coalition = candidate subset satisfying:
1. internal evidence dependency
2. redundant generation influence
3. insufficient external support
```

否则，“联盟发现”实际上只是普通聚类。

---

# 二、Abstract 部分

整体结构很好：

- A1：背景
- A2：威胁
- A3：现有防御缺陷
- A4：提出方法
- A5：CCI
- A6：学习器与安全选择
- A7：实验设置
- A8：实验结果

这是比较标准的 SCI 摘要结构。

## 最大问题：Unsupported Influence 出现前的因果链还不够完整

你在第 5 句中准备写：

> successful attack manifests as evidence coalition that exerts disproportionate influence...

这个判断本身很好。

但是前文需要先解释：

> 为什么 generation influence 可以作为投毒风险的重要依据？

否则审稿人会问：

> Why should influence indicate poisoning?

建议在引言和摘要中补充如下逻辑：

```text
Attack success requires two conditions:

1. malicious documents enter the retrieved context;
2. malicious evidence dominates the generation decision.

Existing defenses mainly detect abnormal documents.

However, coordinated attacks exploit the second property.
```

也就是说，攻击的本质不只是“文档异常”，而是“恶意证据获得了异常控制力”。

你之前 RAG-ConGuard++ 中的攻击成功分解思路可以保留并融入这里，用来支撑 Unsupported Influence 假设。

---

# 三、Introduction 部分

## I3：攻击演进部分

你的规划是：

```text
Zhong 2023
    ↓
One Shot Dominance
    ↓
RIPRAG
```

按攻击能力递进组织相关工作，这个方向是对的。

但是，不建议直接写：

> 攻击已经从单文档发展到联盟攻击。

原因是目前公开研究中，真正明确以“coalition poisoning”命名或形式化的工作可能并不多。

审稿人可能会反驳：

> Existing attacks do not form explicit coalitions.

更稳妥的表述是：

> Existing attacks reveal that multiple poisoned passages can jointly affect retrieval and generation, motivating us to study coordinated poisoning from a set-level perspective.

这样表达的逻辑是：

1. 现有攻击已经展示多文档共同影响；
2. 本文进一步从集合级或联盟级角度进行系统建模；
3. 不强行宣称现有攻击已经正式属于“联盟攻击”。

---

# 四、Methods 总体评价

这是全文最关键的部分。

## M1：Framework Overview

目前七个模块是：

```text
Multi-view Retrieval
        ↓
Claim Extraction
        ↓
Signed Evidence Graph
        ↓
Coalition Discovery
        ↓
Coalitional Counterfactual Influence
        ↓
Risk Learner
        ↓
Safe Context Selection
```

整个流程是完整的，上下游关系也基本合理。

但是，这里存在一个明显问题：

## 模块数量太多，容易给人“七个创新点”的感觉

审稿人会问：

> 这篇论文真正的贡献到底是什么？

建议将贡献明确压缩为三个核心部分。

### 核心贡献 1

**Signed Evidence Graph**

### 核心贡献 2

**Coalitional Counterfactual Influence**

### 核心贡献 3

**Adversary-in-the-loop Training**

其余模块：

- Multi-view Retrieval
- Claim Extraction
- Risk Learner
- Safe Context Selection

应当定位为：

> implementation components 或 supporting modules

而不是每个模块都单独包装成创新点。

你在《新增创新点.md》中已经明确提出：

> 整篇论文最重要的新东西，是中间三个模块。

这个判断是正确的。

---

# 五、M2：多视角检索稳定性

这个模块有价值，但不应强调为独立创新。

原因是：

- 查询改写；
- 稠密检索与 BM25 混合；
- 多视角排名稳定性；

都已经有大量相关研究。

你当前将其定位为“辅助信号”，这个处理是合理的。

Methods 中建议使用类似表述：

> We introduce multi-view retrieval only to reduce retrieval variance and provide auxiliary stability evidence.

不要写成：

> 本文提出一种新的多视角检索稳定性机制。

它更适合作为增强模块和消融变量，而不是核心贡献。

---

# 六、M4：有符号证据图

这是第一个真正有潜力形成方法创新的模块。

但是，目前如果图节点直接定义为 document，表达力度还不够强。

例如：

- 文档 A：The president is Biden.
- 文档 B：The president is Biden.

两个文档可以被判为支持关系。

但是，从方法结构上看，真正发生支持、矛盾和中立关系的对象应该是 **claim**，而不是整篇 document。

你的 M3 已经设计了查询条件主张提取，因此更合理的建模方式是：

```text
图节点：claim
文档：claim 的集合或来源容器
边：claim 之间的支持、矛盾、中立关系
```

这样有三个优势：

1. 支持关系的语义粒度更准确；
2. 同一文档中不同主张可以分别建模；
3. 更容易与已有 document-level NLI graph 方法拉开差异。

否则，ReliabilityRAG 一类工作可能会质疑：

> We also build a document-level relation graph.

---

# 七、M5：证据联盟发现

这是当前设计中风险最大的部分。

目前中文说明中使用：

> support-edge density ≥ τ

来定义或发现联盟。

问题在于，这种方法太简单，容易被认为只是 threshold clustering。

审稿人可能会说：

> This is merely graph clustering based on support edges.

建议至少加入一个更完整的联盟评分或联盟筛选机制，例如：

\[
\operatorname{Score}(C)
=
\alpha \operatorname{Density}(C)
+
\beta \operatorname{CCI}(C)
-
\gamma \operatorname{IndependentSupport}(C).
\]

也就是说，联盟不应仅由图连通性决定，还需要综合：

1. 内部支持关系；
2. 共同生成影响；
3. 外部独立支持不足程度。

目前流程是：

```text
graph
  ↓
cluster
  ↓
CCI
```

这种流程较像普通流水线。

更强的设计是：

```text
graph
  ↓
candidate coalition
  ↓
counterfactual refinement
  ↓
final coalition
```

即先由证据图产生候选联盟，再用 CCI 和独立支持度对候选联盟进行筛选或细化。

这样，CCI 就不只是联盟发现后的一个附加特征，而是联盟定义的一部分。

---

# 八、M6：联盟级反事实生成影响 CCI

这是全文目前最强的部分，也是最接近顶会方法创新的部分。

它解决了传统逐文档 attribution 的核心局限：

```text
remove one document
```

在多篇近重复或互相支持的投毒文档存在时，删除一篇文档后，剩余文档仍然能够维持攻击效果，因此每篇文档的个体影响会被稀释。

你的方法改为：

```text
remove the whole coalition
```

这个思路自然且有明确的问题针对性。

## 数学上需要进一步严谨

你当前计划使用：

\[
\operatorname{CCI}(C)
=
\operatorname{JSD}
\left(
p^{\mathrm{full}}
\parallel
p^{-C}
\right).
\]

这里有一个问题：生成模型输出的是逐 token 的高维概率分布，不能只写成一个模糊的整体分布。

建议明确限定为前 \(H\) 个探针 token，并定义为：

\[
\operatorname{CCI}(C)
=
\frac{1}{H}
\sum_{t=1}^{H}
\operatorname{JSD}
\left(
p_t^{\mathrm{full}}
\parallel
p_t^{-C}
\right).
\]

其中：

\[
p_t^{\mathrm{full}}
=
P_{\theta}
\left(
x_t
\mid
x_{<t},q,D
\right),
\]

\[
p_t^{-C}
=
P_{\theta}
\left(
x_t
\mid
x_{<t},q,D\setminus C
\right).
\]

这样可以明确说明：

1. 比较的是对应生成位置的 token 分布；
2. 使用固定长度的短回答探针控制计算成本；
3. 不需要完整生成长回答；
4. CCI 衡量的是联盟被整体删除后，对生成决策分布造成的平均变化。

---

# 九、M7：分层集合风险学习器

这个部分的研究目的合理，但模型设计有“堆架构”的风险。

目前包括：

- GAT；
- Set Attention；
- 文档级预测；
- 联盟级预测；
- 查询级预测；
- 多层 BCE；
- 排序损失。

审稿人可能会认为：

> The novelty mainly comes from stacking standard neural components.

尤其是 GAT + Set Attention 本身都属于常见模块，如果描述过多，容易掩盖真正的安全洞察。

建议降低模型结构本身的强调程度。

真正应该突出的是：

```text
document risk
coalition risk
query-level attack risk
```

也就是分层监督和集合级风险建模，而不是一定要使用复杂的 GAT。

可以考虑更克制的实现：

- 图消息传递或轻量图编码；
- attention pooling；
- 小型 MLP 输出三级风险。

如果简单模型已经能获得较好结果，反而更容易证明性能来自：

> evidence graph + CCI

而不是大模型容量。

---

# 十、M8：风险约束安全上下文选择

这个模块的方向是合理的。

它将任务从：

> 判断某篇文档是否恶意

升级为：

> 在风险和效用约束下构造一个安全证据子集。

这是一个比较自然的方法闭环。

但是，conformal calibration 需要谨慎使用。

如果直接加入 conformal prediction，却没有：

- 明确的校准集设置；
- 可交换性或覆盖保证讨论；
- 良性误删率的严格实验；
- 不同分布下的校准稳定性分析；

容易被审稿人认为是额外堆叠的热点概念。

你的核心贡献并不是 conformal prediction。

因此建议：

- 第一版先使用验证集校准的风险预算或良性误删预算；
- 如果后续实验充足，再把 conformal calibration 作为增强版本；
- 不要让它成为方法成立的必要条件。

---

# 十一、M9：ADSEA-C 联盟式自适应攻击

这个方向很好，因为它让论文从静态防御扩展到公开防御条件下的自适应攻防。

但是实现难度很高。

ADSEA-C 需要同时优化：

- 多视角可检索性；
- 目标答案诱导；
- 联盟内部支持；
- 低文本重复；
- 与良性证据低冲突；
- 文档级风险；
- 联盟级风险；
- 查询级风险。

这实际上是一个多目标离散优化问题。

你的开发规划预计整个项目需要 16～20 周，这个判断总体合理；其中 ADSEA-C 本身可能占用相当长时间。

建议论文第一版优先保证：

```text
Evidence Graph
+
Coalition Discovery
+
CCI
+
Risk Learner
```

形成完整闭环。

ADSEA-C 应作为强实验和第二条主要贡献，但不要让它阻塞主方法的完成。

---

# 十二、实验设计评价

整体实验设计较完整，也比较接近高水平论文需要的证据链。

其中最重要的是 E8：

> LOO vs CCI

这个实验必须保留。

原因是它直接证明：

> 为什么逐文档归因无法处理冗余协同投毒，以及为什么需要联盟级反事实干预。

建议把 E8 设计成整篇论文的关键诊断实验。

例如设置：

- 注入 1 篇投毒文档；
- 注入 3 篇近重复投毒文档；
- 注入 5 篇互相支持投毒文档。

然后比较：

- 单文档 LOO；
- 单文档 attribution 平均值；
- 整体删除投毒文档；
- 本文 CCI。

预期现象是：

1. 单文档场景下，LOO 和 CCI 都能检测；
2. 冗余文档增加后，单文档 LOO 逐渐下降；
3. 联盟级 CCI 仍保持较高影响分数；
4. 该差距随联盟规模增加而扩大。

这会形成非常直接的方法动机证据。

## E9：与 LLM 安全评估器比较

这个实验可以做，但优先级不高。

将 GPT-4o、Gemini、Claude 等模型作为安全评估器 baseline，可能引出：

- API 版本不一致；
- 提示词敏感；
- 成本不可比；
- 闭源模型更新；
- 延迟测试环境不公平；
- 模型拒答策略影响结果。

因此建议将其放在补充实验或效率分析中，不要作为主实验的核心结论来源。

---

# 最终评分

按照 CCF-B 稿件的标准进行主观评估：

| 部分 | 评价 |
|---|---:|
| 问题定义 | 9/10 |
| 创新性 | 8.5/10 |
| 方法完整性 | 8/10 |
| 理论严谨性 | 7/10 |
| 实验规划 | 9/10 |
| 实现难度 | 9/10 |

综合评价：

> **8.3～8.7 / 10**

如果核心模块能够完整实现，并且实验结果支持联盟级 CCI 的必要性，这个工作具备 CCF-B 竞争力。

---

# 当前最需要修改的五个地方

## 1. 明确 coalition 的数学定义

这是最高优先级。

联盟不能只等于支持边连通分量，应同时包含：

- 内部依赖；
- 共同控制；
- 外部独立支持不足。

否则全文的概念地基不稳。

## 2. 不要把七个模块都包装成创新

建议压缩为三个主要贡献：

1. 有符号证据图与联盟建模；
2. 联盟级反事实生成影响；
3. 攻击者在环的自适应训练。

其余模块作为支撑组件。

## 3. 将 CCI 设为全文绝对核心

全文应围绕：

> 缺乏独立支持的异常生成控制力

展开。

证据图负责提供结构，联盟发现负责提供干预单元，风险学习器负责利用 CCI 完成决策。

## 4. 降低 M7 的模型堆叠感

论文的价值应来自安全问题建模，而不是 GAT、Set Attention 和多损失函数的叠加。

优先使用可解释、轻量的结构。

## 5. 不要让 ADSEA-C 阻塞主线

先完成静态核心方法：

```text
证据图
+
联盟发现
+
CCI
+
风险学习器
+
安全上下文构造
```

再实现 ADSEA-C 和攻击者在环训练。

---

# 总结

总体来说，这版框架比之前的 RAG-ConGuard++ 有明显提升。

之前的版本更像：

> 基于人工统计特征的文档级投毒检测器。

现在的版本已经转向：

> 面向协同知识库投毒的集合级安全机制设计。

当前最大风险已经不是创新不足，而是：

> **创新点太多、概念太满，需要收敛成一个核心科学问题。**

建议将全文核心问题统一为：

> **如何在生成前识别缺乏独立证据支持、却对生成决策产生不成比例控制作用的协同投毒证据集合。**

只要这个主线保持稳定，并通过 LOO 与 CCI 的针对性实验建立清晰证据链，整篇论文的方法逻辑就会更集中、更容易获得审稿人认可。

---

## 参考材料

- 《RAG_论文模仿框架.docx》
- 《新增创新点.md》
- 《新论文时间规划.md》
