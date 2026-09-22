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
para(doc, "RTX PRO 6000 Blackwell(1/4 切片) · NVIDIA T4 · NVIDIA A10 三卡横向对比",
     size=11, color=MUTED, after=2)
para(doc, "测试基础:Triton + TensorRT FP16 · 640×640 · p99 SLA ≤ 50 ms · Azure 按需零售价(2026-09)",
     size=9, color=MUTED, after=10)

# ── 结论摘要 ──
doc.add_heading("结论摘要", level=1)
bullet(doc, "三张卡在 p99 延迟 ≤ 50 ms 的在线推理约束下均可稳定服务,RTX PRO 6000 1/4 切片可用。")
bullet(doc, "按单位推理成本(每百万次推理 USD,越低越省)衡量,RTX PRO 6000 1/4 切片(NC24)最低,约 $1.01。")
bullet(doc, "RTX(NC24)单位成本仅为 T4 的约 65%、A10 的约 32%(即比 T4 省约 35%、比 A10 省约 68%)。")
bullet(doc, "推荐选型:Standard_NC24lds_xl_RTXPRO6000BSE_v6。")

# ── 一、被测配置 ──
doc.add_heading("一、被测配置", level=1)
make_table(doc,
    ["Azure SKU", "GPU · 显存", "vCPU / 内存", "区域", "按需 $/hr"],
    [["Standard_NC24lds_xl_RTXPRO6000BSE_v6", "RTX PRO 6000 (1/4) · 24 GB", "24 / 72 GiB", "West US 2", "1.130"],
     ["Standard_NC36lds_xl_RTXPRO6000BSE_v6", "RTX PRO 6000 (1/4) · 24 GB", "36 / 72 GiB", "West US 2", "1.243"],
     ["Standard_NC16as_T4_v3", "T4 · 16 GB", "16 / 110 GiB", "Sweden Central", "1.276"],
     ["Standard_NV36ads_A10_v5", "A10 · 24 GB", "36 / 440 GiB", "Sweden Central", "4.160"]],
    widths=[2.55, 1.75, 1.05, 1.1, 0.75])
para(doc,
     "注:NC24 与 NC36 是同一块 RTX PRO 6000 1/4 切片(nvidia-smi 确认为真 MIG 实例),仅主机 "
     "vCPU/内存/单价不同;实测在 NC24 上进行,NC36 用同一份吞吐按 $1.243 折算(GPU-bound 吞吐"
     "不随主机 vCPU 增加而下降,该折算偏保守)。",
     size=9, color=MUTED, after=6)

# ── 二、测试方法 ──
doc.add_heading("二、测试方法", level=1)
bullet(doc, "同权重、同 640×640、同 FP16、同 Triton 动态批处理;TensorRT 引擎按卡各自构建(硬件绑定,不可跨卡复用)。")
bullet(doc, "perf_analyzer 并发扫描 1→32,每档先预热再采固定请求数,取 p99;三卡口径一致,保证横向公平。")
bullet(doc, "取每张卡在 p99 ≤ 50 ms 约束内吞吐最高的操作点作为其 QPS@SLA;成本指标 = 单价 ÷ (QPS@SLA × 3600),即每百万次推理成本,越低越省。")

# ── 三、评测结果 ──
doc.add_heading("三、评测结果", level=1)
make_table(doc,
    ["机型 (SKU)", "QPS @ SLA", "p99 (ms)", "并发", "$/hr", "百万次推理成本 ($)"],
    [["NC24lds_xl_RTXPRO6000BSE_v6", "311.2", "44.7", "12", "1.130", "1.01"],
     ["NC36lds_xl_RTXPRO6000BSE_v6", "311.2", "44.7", "12", "1.243", "1.11"],
     ["NC16as_T4_v3", "226.7", "27.5", "6", "1.276", "1.56"],
     ["NV36ads_A10_v5", "362.4", "39.8", "13", "4.160", "3.19"]],
    widths=[2.15, 0.9, 0.75, 0.5, 0.7, 1.15],
    highlight_row=0)
para(doc,
     "百万次推理成本 = 按需单价 ÷ (QPS@SLA × 3600),即 SLA 内每跑 100 万次推理的机时费,越低越省。",
     size=9, color=MUTED, after=2)
para(doc,
     "A10 单机绝对吞吐最高(362 QPS),但单价 $4.16/hr 偏贵,单位推理成本最高;T4 单机吞吐最低。"
     "RTX 1/4 切片以最低单价拿到次高吞吐,单位推理成本最低。",
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
     "已隐含所需机器数量,数值越低越省,与单机吞吐高低无关。", size=9.5, after=4)
make_table(doc,
    ["机型 (SKU)", "百万次推理成本 ($)", "车队 $/hr", "相对 RTX(NC24)"],
    [["NC24lds_xl_RTXPRO6000BSE_v6", "1.01", "36.3", "1.00×(基准)"],
     ["NC36lds_xl_RTXPRO6000BSE_v6", "1.11", "39.9", "1.10×"],
     ["NC16as_T4_v3", "1.56", "56.3", "1.55×"],
     ["NV36ads_A10_v5", "3.19", "114.8", "3.16×"]],
    widths=[2.4, 1.25, 1.05, 1.4],
    highlight_row=0)

# ── 五、结论与建议 ──
doc.add_heading("五、结论与建议", level=1)
bullet(doc, "首选 Standard_NC24lds_xl_RTXPRO6000BSE_v6(RTX PRO 6000 1/4 切片):满足 50 ms SLA,成本效率最优,同等预算下车队吞吐最高。")
bullet(doc, "若主机侧需要更多 vCPU/内存,可选 NC36 同切片方案,成本效率仍优于 T4 与 A10。")
bullet(doc, "A10 适合追求单机绝对吞吐、对单价不敏感的场景;T4 在本模型下无成本或性能优势。")

# ── 六、测试范围与说明 ──
doc.add_heading("六、测试范围与说明", level=1)
bullet(doc, "价格为 Azure 公开零售价(Consumption / 按需 / Linux / 非 Spot,查询于 2026-09);如有 EA/预留折扣,按实际单价折算即可,不影响相对结论。", size=9.5)
bullet(doc, "本报告仅纳入实测机型与向更多 vCPU 的保守折算(RTX NC36);未纳入向更少 vCPU 主机(如 NC4as/NC8as_T4_v3)的推算,以保证口径一致。", size=9.5)
bullet(doc, "测试针对模型前向吞吐(不烘焙 NMS),对加速器选型与成本这是核心差异项;显存对本模型/batch 三卡均不构成瓶颈。", size=9.5)

doc.save("report.docx")
print("已生成 report.docx")
