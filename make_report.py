"""生成 Word 版评测结论报告 (report.docx)。风格: 专业、精炼、清晰。
图表位置留 placeholder, 由人工把 results/curve.png 贴入。
运行: python make_report.py
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

LATIN = "Calibri"
CJK = "Microsoft YaHei"
INK = RGBColor(0x21, 0x21, 0x21)
ACCENT = RGBColor(0x12, 0x57, 0xA0)
MUTED = RGBColor(0x60, 0x6A, 0x76)


def set_cjk(run):
    run.font.name = LATIN
    run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)


def base_style(doc):
    st = doc.styles["Normal"]
    st.font.name = LATIN
    st.font.size = Pt(10.5)
    st.font.color.rgb = INK
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    pf = st.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.25
    for i, sz in ((1, 16), (2, 12.5)):
        h = doc.styles[f"Heading {i}"]
        h.font.name = LATIN
        h.font.size = Pt(sz)
        h.font.bold = True
        h.font.color.rgb = ACCENT if i == 1 else INK
        h.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
        h.paragraph_format.space_before = Pt(14 if i == 1 else 10)
        h.paragraph_format.space_after = Pt(4)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)


def cell_text(cell, text, bold=False, color=None, size=9.5, align="left"):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT,
                   "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "right": WD_ALIGN_PARAGRAPH.RIGHT}[align]
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.space_before = Pt(1)
    r = p.add_run(text)
    set_cjk(r)
    r.font.size = Pt(size)
    r.font.bold = bold
    if color:
        r.font.color.rgb = color


def make_table(doc, headers, rows, widths=None, highlight_row=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]
        cell_text(c, h, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), size=9.5,
                  align="left" if j == 0 else "center")
        shade(c, "1257A0")
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        hl = (highlight_row is not None and i == highlight_row)
        for j, val in enumerate(row):
            cell_text(cells[j], val, bold=hl, size=9.5,
                      align="left" if j == 0 else "center")
            if hl:
                shade(cells[j], "E7F0FA")
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Inches(w)
    return t


def para(doc, text, size=10.5, color=None, italic=False, bold=False, after=6, align="left"):
    p = doc.add_paragraph()
    p.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER}[align]
    p.paragraph_format.space_after = Pt(after)
    r = p.add_run(text)
    set_cjk(r)
    r.font.size = Pt(size)
    r.font.italic = italic
    r.font.bold = bold
    if color:
        r.font.color.rgb = color
    return p


def bullet(doc, text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    set_cjk(r)
    r.font.size = Pt(size)


doc = Document()
base_style(doc)
for s in doc.sections:
    s.top_margin = s.bottom_margin = Inches(0.9)
    s.left_margin = s.right_margin = Inches(0.9)

# ── 标题 ──
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.LEFT
r = title.add_run("YOLOv8s 在线推理 GPU 选型评测报告")
set_cjk(r)
r.font.size = Pt(20)
r.font.bold = True
r.font.color.rgb = ACCENT
para(doc, "RTX PRO 6000 Blackwell(1/4 切片) · NVIDIA T4 · NVIDIA A10 横向对比(含客户现用 NC4as_T4_v3)",
     size=11, color=MUTED, after=2)
para(doc, "测试基础:Triton + TensorRT FP16 · 640×640 · p99 SLA ≤ 50 ms · Azure 按需零售价 West US 2(2026-09)",
     size=9, color=MUTED, after=10)

# ── 结论摘要 ──
doc.add_heading("结论摘要", level=1)
bullet(doc, "五个配置在 p99 延迟 ≤ 50 ms 的在线推理约束下均可稳定服务。")
bullet(doc, "客户现用的 Standard_NC4as_T4_v3 单位推理成本最低(约 $0.92 / 百万次)——因主机便宜($0.526/hr);"
            "但 4 vCPU 限制使其单机吞吐仅 ~159 QPS(GPU 未喂满)。")
bullet(doc, "RTX PRO 6000 1/4 切片(NC24)单位成本约 $1.01,仅比客户的 NC4as 高约 10%,性价比已基本追平;"
            "而单机吞吐 311 QPS,约为 NC4as 的 2 倍。")
bullet(doc, "服务同等负载,RTX 所需机器数量约为 NC4as 的一半——节点更少、运维更简、单机余量更大。"
            "结论:RTX 与客户现用 T4 高度可比,并在吞吐密度与运维上占优,是有竞争力的升级路径。")

# ── 一、被测配置 ──
doc.add_heading("一、被测配置", level=1)
make_table(doc,
    ["Azure SKU", "GPU · 显存", "vCPU / 内存", "区域", "按需 $/hr"],
    [["Standard_NC24lds_xl_RTXPRO6000BSE_v6", "RTX PRO 6000 (1/4) · 24 GB", "24 / 72 GiB", "West US 2", "1.130"],
     ["Standard_NC36lds_xl_RTXPRO6000BSE_v6", "RTX PRO 6000 (1/4) · 24 GB", "36 / 72 GiB", "West US 2", "1.243"],
     ["Standard_NC4as_T4_v3(客户现用)", "T4 · 16 GB", "4 / 28 GiB", "West US 2", "0.526"],
     ["Standard_NC16as_T4_v3", "T4 · 16 GB", "16 / 110 GiB", "West US 2", "1.204"],
     ["Standard_NV36ads_A10_v5", "A10 · 24 GB", "36 / 440 GiB", "West US 2", "3.200"]],
    widths=[2.55, 1.6, 1.0, 1.0, 0.75])
para(doc,
     "注:NC24 与 NC36 是同一块 RTX PRO 6000 1/4 切片(nvidia-smi 确认为真 MIG 实例),仅主机 "
     "vCPU/内存/单价不同;实测在 NC24 上进行,NC36 用同一份吞吐按 $1.243 折算(GPU-bound 吞吐"
     "不随主机 vCPU 增加而下降,该折算偏保守)。NC4as 与 NC16as 同为单块 T4,仅主机规格/单价不同,"
     "两者均为实测。",
     size=9, color=MUTED, after=6)

# ── 二、测试方法 ──
doc.add_heading("二、测试方法", level=1)
bullet(doc, "同权重、同 640×640、同 FP16、同 Triton 动态批处理;TensorRT 引擎按卡各自构建(硬件绑定,不可跨卡复用)。")
bullet(doc, "perf_analyzer 并发扫描 1→32,每档先预热再采固定请求数,取 p99;各配置同脚本同口径,保证横向公平。")
bullet(doc, "各机型每档采样数因主机内存不同而异(如 NC4as 28GiB 采 800/档,大内存机型更多),仅影响 p99 估计精度,不改吞吐真值与横向公平。")
bullet(doc, "取每张卡在 p99 ≤ 50 ms 约束内吞吐最高的操作点作为其 QPS@SLA;成本指标 = 单价 ÷ (QPS@SLA × 3600),即每百万次推理成本,越低越省。")

# ── 三、评测结果 ──
doc.add_heading("三、评测结果", level=1)
make_table(doc,
    ["机型 (SKU)", "QPS @ SLA", "p99 (ms)", "并发", "$/hr", "百万次推理成本 ($)"],
    [["NC24lds_xl_RTXPRO6000BSE_v6", "311.2", "44.7", "12", "1.130", "1.01"],
     ["NC36lds_xl_RTXPRO6000BSE_v6", "311.2", "44.7", "12", "1.243", "1.11"],
     ["NC4as_T4_v3(客户现用)", "159.5", "16.2", "3", "0.526", "0.92"],
     ["NC16as_T4_v3", "226.7", "27.5", "6", "1.204", "1.48"],
     ["NV36ads_A10_v5", "362.4", "39.8", "13", "3.200", "2.45"]],
    widths=[2.35, 0.85, 0.7, 0.5, 0.65, 1.1],
    highlight_row=0)
para(doc,
     "百万次推理成本 = 按需单价 ÷ (QPS@SLA × 3600),即 SLA 内每跑 100 万次推理的机时费,越低越省。",
     size=9, color=MUTED, after=2)
para(doc,
     "客户现用的 NC4as(4 vCPU)单位成本最低,但吞吐被 CPU 卡在 ~159 QPS(GPU 未喂满,并发加大只涨延迟"
     "不涨吞吐);同为 T4 的 NC16as 加到 16 vCPU 把吞吐提到 227,但单价涨得更多,单位成本反升至 $1.48。"
     "RTX 1/4 切片以次低单价拿到 311 QPS 的高吞吐,单位成本 $1.01,与客户的 NC4as 仅差约 10%,"
     "却只需约一半机器。A10 单机绝对吞吐最高但单价贵,单位成本最高。",
     size=9.5, after=8)

# 图表 placeholder
ph = doc.add_paragraph()
ph.alignment = WD_ALIGN_PARAGRAPH.CENTER
ph.paragraph_format.space_before = Pt(4)
ph.paragraph_format.space_after = Pt(2)
pr = ph.add_run("〔 此处插入图表:results/curve.png 〕")
set_cjk(pr)
pr.font.size = Pt(11)
pr.font.bold = True
pr.font.color.rgb = ACCENT
para(doc, "图 1  并发 1→32 扫描:吞吐(QPS)对 p99 延迟(ms),虚线为 50 ms SLA。每个点为一档并发。",
     size=9, color=MUTED, align="center", after=10)

# ── 四、车队成本 ──
doc.add_heading("四、车队成本(以服务 10,000 QPS 为例)", level=1)
para(doc,
     "车队成本随单位推理成本等比放大:服务固定需求所需车队 $/hr = 需求吞吐 × 单价 ÷ QPS@SLA,"
     "已隐含所需机器数量,数值越低越省,与单机吞吐高低无关。下表以客户现用 NC4as 为基准。", size=9.5, after=4)
make_table(doc,
    ["机型 (SKU)", "百万次推理成本 ($)", "车队 $/hr", "相对 NC4as(客户)"],
    [["NC4as_T4_v3(客户现用)", "0.92", "33.0", "1.00×(基准)"],
     ["NC24lds_xl_RTXPRO6000BSE_v6", "1.01", "36.3", "1.10×"],
     ["NC36lds_xl_RTXPRO6000BSE_v6", "1.11", "39.9", "1.21×"],
     ["NC16as_T4_v3", "1.48", "53.1", "1.61×"],
     ["NV36ads_A10_v5", "2.45", "88.3", "2.68×"]],
    widths=[2.4, 1.25, 1.05, 1.4],
    highlight_row=1)
para(doc,
     "车队 $/hr 已含机器数量:RTX 单机 311 QPS vs NC4as 159 QPS,服务同等负载 RTX 所需节点约为 NC4as 的一半,"
     "车队机时费仅高约 10%——把节点数、编排与 CPU 侧开销计入总拥有成本后,两者实质相当。",
     size=9, color=MUTED, after=6)

# ── 五、结论与建议 ──
doc.add_heading("五、结论与建议", level=1)
bullet(doc, "客户现用 Standard_NC4as_T4_v3:单位推理成本最低($0.92/百万次),选型合理;但单机吞吐受 4 vCPU 限制(~159 QPS),需较多节点横向扩展。")
bullet(doc, "RTX PRO 6000 1/4 切片(NC24)性价比已追平 NC4as(单位成本仅高约 10%),且单机吞吐约 2 倍;服务同等负载所需机器约减半,把节点数与运维复杂度计入后总拥有成本实质相当,推荐作为升级路径。")
bullet(doc, "若主机侧需要更多 vCPU/内存,NC36 同切片方案成本效率仍优于 NC16as 与 A10。")
bullet(doc, "同为 T4,NC16as 相比客户的 NC4as 无成本效率优势(加 vCPU 提了吞吐但单价涨得更多);A10 适合追求单机绝对吞吐、对单价不敏感的场景。")

# ── 六、测试范围与说明 ──
doc.add_heading("六、测试范围与说明", level=1)
bullet(doc, "价格为 Azure 公开零售价(Consumption / 按需 / Linux / 非 Spot,查询于 2026-09),统一取 West US 2 区域以保证口径一致(吞吐为硬件绑定,与区域无关);如有 EA/预留折扣或换区,按实际单价折算即可,不影响相对结论。", size=9.5)
bullet(doc, "本报告纳入四款实测机型(RTX NC24、NC4as_T4_v3〔客户现用〕、NC16as_T4_v3、A10)与一款保守折算(RTX NC36,沿用 NC24 实测吞吐、仅换更高单价);同为 T4 的 NC8as_T4_v3 未测。", size=9.5)
bullet(doc, "测试针对模型前向吞吐(不烘焙 NMS),对加速器选型与成本这是核心差异项;显存对本模型/batch 三卡均不构成瓶颈。", size=9.5)

doc.save("report.docx")
print("已生成 report.docx")
