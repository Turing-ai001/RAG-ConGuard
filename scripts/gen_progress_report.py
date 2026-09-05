# -*- coding: utf-8 -*-
"""生成 RAG-ConGuard 实质性进展汇报 Word 文档（2026-09-01）。

所有数字来源：experiments/final_protocol/FINAL_EXPERIMENT_REPORT.md（冻结协议 v1.2 真实产出）。
用法：python scripts/gen_progress_report.py
输出：进度汇报/进展汇报_RAG-ConGuard_2026-09-01.docx
"""
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "进度汇报", "进展汇报_RAG-ConGuard_2026-09-01.docx")

GRAY = "D9D9D9"       # 本文方法行底色
BLUE = "2E74B5"       # 标题色
BLACK = "000000"
DARK = "1A1A1A"


def set_font(run, name_cn="宋体", name_en="Calibri", size=11, bold=False, color=None):
    run.font.name = name_en
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), name_cn)
    rFonts.set(qn("w:ascii"), name_en)
    rFonts.set(qn("w:hAnsi"), name_en)


def para(doc, text="", size=11, bold=False, cn="宋体", en="Calibri",
         color=None, align=None, space_after=6, first_indent=True):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.35
    if align:
        p.alignment = align
    if first_indent and text:
        pf.first_line_indent = Pt(22)
    r = p.add_run(text)
    set_font(r, name_cn=cn, name_en=en, size=size, bold=bold, color=color)
    return p


def heading(doc, text, level=1):
    sizes = {1: 14, 2: 12, 3: 11}
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(14 if level == 1 else 10)
    pf.space_after = Pt(6)
    r = p.add_run(text)
    set_font(r, name_cn="微软雅黑", name_en="Calibri", size=sizes[level],
             bold=True, color=BLUE)
    return p


def bullet(doc, text, bold_prefix=None, size=11):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(4)
    pf.line_spacing = 1.3
    pf.left_indent = Pt(16)
    if bold_prefix:
        r1 = p.add_run(bold_prefix)
        set_font(r1, name_cn="宋体", size=size, bold=True)
    r = p.add_run(text)
    set_font(r, name_cn="宋体", size=size)
    return p


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def set_table_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "BFBFBF")
        borders.append(el)
    tblPr.append(borders)


def table(doc, headers, rows, hl_rows=(), col_align=None, font_size=9.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    set_table_borders(t)
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        set_font(r, name_cn="微软雅黑", size=font_size, bold=True, color=BLACK)
        shade(hdr[i], "EFEFEF")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            cells[ci].text = ""
            p = cells[ci].paragraphs[0]
            if col_align is None or ci >= len(col_align):
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = col_align[ci]
            r = p.add_run(str(val))
            set_font(r, name_cn="宋体", size=font_size)
            if ri in hl_rows:
                shade(cells[ci], GRAY)
    return t


def note(doc, text, size=9):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(8)
    r = p.add_run(text)
    set_font(r, name_cn="宋体", size=size, color="595959")
    return p


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.3)
    sec.left_margin = sec.right_margin = Cm(2.5)

    # ---------- 标题区 ----------
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("RAG-ConGuard 项目实质性进展汇报")
    set_font(r, name_cn="微软雅黑", size=18, bold=True, color=BLACK)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("面向协同投毒的联盟感知反事实防御（RAG 安全方向）")
    set_font(r, name_cn="微软雅黑", size=11, bold=False, color="595959")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("汇报人：锦松   ·   2026-09-01   ·   状态：实验收尾完成，论文进入对齐改写与投稿阶段")
    set_font(r, name_cn="宋体", size=9.5, color="595959")

    # ---------- 总览 ----------
    para(doc, "核心假设（Unsupported Influence）：协同投毒不是一个「可疑文档」，而是一个证据联盟——内部互相支持、"
              "外部缺乏独立佐证、却对生成结果有不成比例的控制力。本阶段完成了系统实现、实验体系审计与重建、"
              "全部对照实验补齐，以及论文初稿，得到一组可信、可复现的主结果。", size=11, first_indent=True)

    table(doc, ["核心指标", "数值", "说明"],
          [["无防御基线 ASR（攻击成功率）", "0.833", "冻结 test n=60，三数据集（NQ/HotpotQA/MS MARCO）"],
           ["本文方法·可用性操作点 G1", "0.583", "攻击成功率降至 0.583，拒答仅 6.7%，准确率 0.100→0.300"],
           ["本文方法·强保护操作点 G0", "0.217", "攻击成功率压至 0.217，代价是 40% 拒答"],
           ["自适应攻击逃脱率", "0", "3 轮攻击者在环 + 从未参与的未见变体族，均未逃过；检测器 AUC 0.865"],
           ["干净查询效用损失", "+2~3pp", "良性误滤 8.7%–13.5%，无防御本身自带 28.6% 拒答基线"]],
          hl_rows=(1, 2), font_size=9.5)

    # ---------- 一、五项实质工作 ----------
    heading(doc, "一、本阶段完成的五项实质工作")
    para(doc, "1. 全链路防御系统实现（6 个模块贯通）", bold=True, size=11, first_indent=False)
    para(doc, "多视角检索（BGE 向量 + BM25，RRF 融合 top-5）→ Claim 表示 → 有符号证据图（NLI 三分类阈值 0.7 "
              "+ 数值冲突启发式 + LLM 兜底 α=0.3）→ 联盟发现（正边连通分量 + 精化 v1.2：size≥2 且凝聚力/影响力为正，"
              "孤立单点豁免）→ 联盟级反事实影响度量（信任传播 + 答案支持 + target-free 候选，消除推理期 oracle 泄漏）"
              "→ 风险约束安全上下文选择（整联盟剔除 / 仅删最高风险联盟 / 贪心重建三种策略）。攻击者在环训练贯穿后三个模块，"
              "全部中间产物落盘，在 AutoDL 4090 上可复现运行。")
    para(doc, "2. 实验体系全面审计与重建（本轮最关键的进展）", bold=True, size=11, first_indent=False)
    para(doc, "对既有实验做全量审计，识别 7 个硬伤并逐一定位根因。其中最致命的问题：旧结果「低攻击成功率」是靠拒答率 100% "
              "换来的——防御在所有查询上直接弃答，等于没有防御能力。根因定位到两个具体缺陷并已修复：")
    bullet(doc, "：原用 purity=max(投毒比,良性比) 做正例标签，全良性联盟也被标为正（训练集仅 8 个负例），学习器对一切"
                "输出高分→任意阈值下全拒答。已改为 poisoned_ratio≥0.8 标签 + 干净查询联盟作负例池（负例 8→2843）。",
           bold_prefix="① 标签定义错误")
    bullet(doc, "：整联盟剔除策略在 top-5 被同一联盟控制时会清空全部上下文。新增 S2 贪心重建（保留下限 λ）。",
           bold_prefix="② 选择器无保留删除")
    para(doc, "修复后 val AUC 0.769→0.858，τ=0.05 时投毒联盟漏检=0。随即冻结实验协议 v1.2：统一三层 split（投毒 "
              "300 查询按 60/20/20 分层 × 3 数据集，干净集 200/100/300）、逐查询互斥状态标注（攻击成功/正确/拒答/其他）、"
              "配对 bootstrap（1000 次）统计检验。旧协议产物全部作废，当前所有结论均来自冻结协议。")
    para(doc, "3. 强基线与对照实验补齐", bold=True, size=11, first_indent=False)
    para(doc, "新增 TrustRAG 原版公平复刻（固定 commit，BGE 替代 Contriever）、逐文档 LOO 防御（B6）、NLI 一致性防御"
              "（B4，ReliabilityRAG 风格），与本文方法共用同检索器、同语料、同 split、同生成模型；补干净查询效用实验"
              "（clean_test 900 条）；补跨大语言模型（Qwen2.5-7B / Llama-3.1-8B / Mistral-7B-v0.3）；补 3 轮自适应攻防"
              " + 3 组从未参与训练的未见变体族。")
    para(doc, "4. 机制诊断实验 E8：冗余投毒下的稀释效应", bold=True, size=11, first_indent=False)
    para(doc, "核心动机「逐文档归因在协调投毒下失效」已获得直接数量证据（见第二部分 E8 表）：组规模 2/3/4 时单篇删减影响"
              "慢增（0.353→0.521），联盟级删减快增（0.480→0.944），差距 1.36×→1.81× 单调扩大。")
    para(doc, "5. 论文初稿与配套产出", bold=True, size=11, first_indent=False)
    para(doc, "英文论文初稿 v4（46 张卡片正文 + 4 张真实数据表，表注齐全）已成形，审阅意见 8 条全部落实；配套产出："
              "方法框架图/场景图（fig1–fig2 已就绪，fig3–fig4 待数据对齐后重制）、参考文献库（9 篇已验证 + 8 篇待核）、"
              "中国计算机学会 B 类与 SCI 二区双轨定位文档。")

    # ---------- 二、主结果 ----------
    heading(doc, "二、主实验结果")
    para(doc, "测试集 n=60（每数据集 20 条，三层冻结 split），逐查询互斥四态判定，配对 bootstrap 1000 次。攻击面：检索 "
              "asr@5=1.00，top-5 平均含 3.82 篇投毒文档——这是极端攻击密度（语料注入率约 0.08%）。")
    table(doc, ["方法", "类型", "ASR↓", "准确率↑", "拒答率", "ΔASR vs 无防御"],
          [["B4 NLI 一致性", "发布级", "0.867", "0.083", "0.033", "+0.033"],
           ["B0 无防御", "基线", "0.833", "0.100", "0.017", "—"],
           ["B1 相关性过滤", "逐文档", "0.833", "0.100", "0.017", "0.000"],
           ["B2 重复过滤", "逐文档", "0.750", "0.200", "0.017", "−0.083"],
           ["B6 逐文档 LOO", "逐文档", "0.650", "0.217", "0.067", "−0.183"],
           ["G1 可用性点（本文）", "联盟级", "0.583", "0.300", "0.067", "−0.250"],
           ["G0 强保护点（本文）", "联盟级", "0.217", "0.333", "0.400", "−0.617"],
           ["B5 TrustRAG 原版复刻", "发布级", "0.150", "0.333", "0.483", "−0.683"]],
          hl_rows=(5, 6), font_size=9.5)
    note(doc, "注：95% 置信区间（配对 bootstrap，vs B0）：G1 Δ=−0.250，CI [−0.367, −0.133]；G0 Δ=−0.617，CI [−0.717, −0.517]。两者均不跨 0。")
    para(doc, "三个必须一起讲的结论：", bold=True, size=11, first_indent=False)
    bullet(doc, "在拒答 6.7% 这一层，G1（0.583）优于逐文档 LOO（0.650）与重复过滤（0.750），且准确率同时从 0.100 提升到 0.300。",
           bold_prefix="① 同拒答层位下，联盟级优于逐文档：")
    bullet(doc, "NLI 一致性基线 ASR 0.867，反而高于无防御的 0.833——「一致的多数簇」恰恰就是投毒簇，多数一致性推理在协同投毒下会主动帮攻击者。这是本文动机最强的反例证据。",
           bold_prefix="② B4 倒挂，反向印证问题选得对：")
    bullet(doc, "B5 的 ASR 0.150 低于 G0 的 0.217，但 B5 拒答更高（48.3% vs 40.0%），且其「防御」是让 LLM 自评估后直接改写答案（证据级入侵、不可审计），G0 是纯证据选择（可审计、可追溯）。该对照正面呈现，不回避。",
           bold_prefix="③ 与 TrustRAG 同层，但机制不同：")

    # ---------- 三、支撑性证据 ----------
    heading(doc, "三、支撑性证据")
    heading(doc, "E8 稀释机制（核心动机，target-free 口径，test 39 组 ≥2 篇投毒组）", level=2)
    table(doc, ["投毒组规模", "组数", "逐文档 LOO", "联盟级删除", "比值"],
          [["2 篇", "15", "0.353", "0.480", "1.36×"],
           ["3 篇", "8", "0.466", "0.714", "1.53×"],
           ["4 篇", "16", "0.521", "0.944", "1.81×"]], font_size=9.5)
    note(doc, "注：组规模增大时单篇删除的影响被剩余成员稀释（慢增），联盟级删除的控制力快速放大，差距随规模单调扩大——「逐文档归因无法处理冗余协同投毒」的直接量化证据。")

    heading(doc, "跨大语言模型泛化（同 split、同参数）", level=2)
    table(doc, ["生成模型", "无防御 ASR", "G0 强保护点", "G1 可用性点", "干净侧误滤"],
          [["Qwen2.5-7B", "0.833", "0.217 @ 40.0%", "0.583 @ 6.7%", "13.5%"],
           ["Llama-3.1-8B", "0.650", "0.250 @ 41.7%", "0.550 @ 11.7%", "7.1%"],
           ["Mistral-7B-v0.3", "0.883", "0.300 @ 33.3%", "0.650 @ 0.0%", "7.1%"]], font_size=9.5)
    note(doc, "注：三个不同模型族的防御方向完全一致——强保护点压至 0.25–0.30，可用性点 0.55–0.65 且拒答极低。附带发现：Llama 天生更抗定向投毒（0.650），Mistral 最易被攻击（0.883）。")

    heading(doc, "自适应攻防（攻击者在环，3 轮 + 未见变体族）", level=2)
    table(doc, ["轮次", "检测器", "val AUC", "逃脱率 a / b / c"],
          [["round 0", "规则", "0.886", "0.00 / 0.00 / 0.00"],
           ["round 1", "MLP", "0.812", "0.03 / 0.00 / 0.07"],
           ["round 2", "MLP", "0.837", "0.00 / 0.00 / 0.00"],
           ["未见变体族 d/e/f", "最终 MLP", "0.865", "0.00 / 0.00 / 0.00"]],
          hl_rows=(3,), font_size=9.5)
    note(doc, "注：训练中见过的变体与完全未参与训练的 d/e/f 族，逃脱率均为 0；检测器 AUC 维持在 0.81–0.87。")

    heading(doc, "干净查询效用（clean_test 900 条）", level=2)
    para(doc, "无防御 Vanilla 本身自带 28.6% 拒答（模型弃答基线）。防御的边际代价：G0 拒答 +3.3pp、良性误滤 8.7%；"
              "G1 拒答 +2.2pp、良性误滤 13.5%。效用保留良好（满足误滤 ≤15% 的协议约束）。")

    # ---------- 四、限制 ----------
    heading(doc, "四、如实说明的三条限制（全部如实报告，不筛数）")
    bullet(doc, "论文原先将影响形式化为生成分布的 JSD，实现用的是「答案支持分数」近似。已补做精确版本（teacher-forced 对齐前缀 + 逐位置 JSD），与代理的 Spearman 相关在 H=8/16/32 下稳定为 0.36/0.39/0.39。结论：代理是有效的排序信号，但不是强代理——论文不得声称「代理=JSD」，将改写为「证据侧近似 + 精确 CCI 对照分析（局限性栏）」。",
           bold_prefix="① 代理 CCI 与精确 CCI 仅中等相关（ρ≈0.39）：")
    bullet(doc, "RuleRisk 规则与 MLP 学习器端到端对比（λ=0.2）：0.533/0.350 vs 0.583/0.300，规则略优。学习器价值体现在判别 AUC（0.79–0.86 vs 规则 0.32 系）与阈值自动化，而非端到端增益。如实写入，不拔高。",
           bold_prefix="② 规则风险在端到端指标上略优于 MLP：")
    bullet(doc, "本设置下 58% 的查询整个 top-5 被同一联盟控制，此时拒答是正确动作而非缺陷。因此论文给出两个操作点（G0 强保护 / G1 可用性）而非单一最优。另：BEIR 当前快照不含 gold answer 文件，干净查询的答案准确率不可得，效用只能以误滤率为代理——限制如实标注。",
           bold_prefix="③ 高攻击密度下「强保护」与「低拒答」不可兼得：")

    # ---------- 五、下一步 ----------
    heading(doc, "五、下一步与时间线（目标：10 月前完成投稿）")
    para(doc, "当前最重要的待办——论文与实验对齐：英文初稿写于最终数据产出之前，核心数字仍来自已作废的旧协议，且方法章节存在"
              "一处推理期 oracle 泄漏表述（候选答案写法与摘要声称的 target-free 矛盾）。为纯文字改写工作，1–2 天可完成，"
              "无需重跑任何实验。", size=11)
    table(doc, ["时间", "任务", "内容"],
          [["第 1 周 9/01–9/07", "论文—实验对齐改写",
            "按冻结协议重写摘要与实验章节全部数字；修正方法章节 oracle 泄漏表述、联盟定义、学习器定位；重做 fig3/fig4"],
           ["第 2 周 9/08–9/14", "定刊与排版",
            "确定 2–3 个 SCI 二区候选期刊并比对 scope 与页数限制；按目标模板排版；精修 Introduction 与 Related Work；核对参考文献"],
           ["第 3 周 9/15–9/21", "内部审读",
            "请导师/同学通读一轮，收集意见修改；图表转高清矢量；术语与重复率检查"],
           ["第 4 周 9/22–9/30", "终稿与提交",
            "语言润色、格式终检、cover letter、补充材料整理，完成投稿"]], font_size=9.5)
    para(doc, "需要老师决策/把关的两点：", bold=True, size=11, first_indent=False)
    bullet(doc, "目前方向是 SCI 二区，具体刊物未定，希望老师推荐或帮助缩小范围。", bold_prefix="① 目标期刊选择：")
    bullet(doc, "论文是双点并列呈现（强保护 + 可用性），还是以可用性点为主、强保护点放附录？这会影响审稿人对「实用性」的观感。",
           bold_prefix="② 操作点取舍：")

    # ---------- 页脚说明 ----------
    note(doc, "数据来源：experiments/final_protocol/results/（冻结协议 v1.2，AutoDL 4090 真实产出）；权威解读：FINAL_EXPERIMENT_REPORT.md；审计记录：PROJECT_AUDIT.md。本报告不筛数、不修饰，负结果与限制全部保留。")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc.save(OUT)
    print("已生成:", OUT)


if __name__ == "__main__":
    main()
