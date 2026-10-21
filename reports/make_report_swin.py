"""生成三期 Swin Transformer FP8 量化分析报告 (report_swin.docx)。
数据来源: results/quant/rtx6000-quarter-nc24-trt{10,11}-swin*, 逐层分类见报告 3.2 节说明。
运行: python reports/make_report_swin.py
"""
from pathlib import Path

from report_style import MUTED, bullet, make_table, new_doc, para

doc = new_doc(
    "Swin Transformer FP8 量化推理性能分析报告",
    "RTX PRO 6000 Blackwell(1/4 切片) · Swin-T · FP16 vs FP8 · TensorRT 11.3 / 10.13",
    "测试基础:Standard_NC24lds_xl_RTXPRO6000BSE_v6 · Swin-T 224×224(官方 ImageNet-22K 权重)· NVIDIA ModelOpt PTQ · 2026-10")


def note(text):
    para(doc, text, size=9, color=MUTED, after=6)


# ── 结论摘要 ──
doc.add_heading("结论摘要", level=1)
bullet(doc, "复现客户现象:ModelOpt 默认 FP8 相比 FP16 仅提升 20%(batch 8),TensorRT 10.13 为 16%。")
bullet(doc, "根因一:FP8 只加速线性层(1.45×);LayerNorm、注意力、窗口重排、分类头约占 GPU 时间 38%,不受益。")
bullet(doc, "根因二:FP8 下 GELU 与前一层 GEMM 的融合断开,变为独立 kernel,抵消约 40% 的 GEMM 收益。")
bullet(doc, "两项改动后提升至 1.37×(batch 8)/ 1.56×(batch 32):GELU 改 tanh 近似导出 + 分类头一并量化,精度不变。")
bullet(doc, "注意力 FP8 量化几乎无收益;NVIDIA 示例的 torch 量化路径反而比 FP16 更慢。")

# ── 一、测试环境 ──
doc.add_heading("一、测试环境", level=1)
make_table(doc, ["项目", "配置"], [
    ["Azure VM", "Standard_NC24lds_xl_RTXPRO6000BSE_v6(24 vCPU / 72 GiB)"],
    ["GPU", "RTX PRO 6000 Blackwell 1/4 MIG 切片 · 24 GB · 驱动 595.58"],
    ["推理栈", "Triton 26.09(TensorRT 11.3.0,当前最新)/ Triton 25.08(TensorRT 10.13.2)"],
    ["模型", "Swin-T(swin_tiny_patch4_window7_224_22k,官方权重,21841 类)· timm 加载 · ONNX opset 17 · 动态 batch 1–32"],
    ["量化工具", "NVIDIA ModelOpt 0.47.0"],
    ["数据", "COCO val2017:500 张校准 + 4500 张评测(不重叠)"],
], widths=[1.3, 5.4])

# ── 二、测试方法 ──
doc.add_heading("二、测试方法", level=1)
make_table(doc, ["环节", "方法"], [
    ["FP8 量化", "ModelOpt ONNX PTQ · E4M3 · max 校准 · 默认量化 MatMul / Gemm"],
    ["FP16 基线", "TRT 11:AutoCast FP16 ONNX;TRT 10:FP32 ONNX + --fp16"],
    ["性能", "trtexec GPU 纯计算吞吐 · batch 1 / 8 / 32 · 每点 15 s"],
    ["耗时归因", "trtexec 逐层 profile(batch 8),按算子类别汇总"],
    ["精度", "Triton 加载实际引擎 · 4500 张 · 与 FP16 引擎输出的 top-1 一致率(22K 无官方验证集,只作相对比较)"],
], widths=[1.3, 5.4])

# ── 三、测试结果 ──
doc.add_heading("三、测试结果", level=1)

doc.add_heading("3.1 默认 FP8 vs FP16(GPU 纯计算吞吐 img/s,括号内为相对 FP16)", level=2)
make_table(doc, ["配置", "batch 1", "batch 8", "batch 32"], [
    ["TRT 10.13 · FP16", "990", "3,820", "3,497"],
    ["TRT 10.13 · FP8", "1,214(1.23×)", "4,437(1.16×)", "4,220(1.21×)"],
    ["TRT 11.3 · FP16", "931", "3,685", "3,322"],
    ["TRT 11.3 · FP8", "1,213(1.30×)", "4,428(1.20×)", "4,233(1.27×)"],
], widths=[1.8, 1.4, 1.4, 1.4], highlight_row=3)
note("与客户反馈的 10%–20% 一致。两个 TensorRT 版本的 FP8 吞吐相同,版本不是 Swin 的主要因素(与 YOLOv8 不同)。")

doc.add_heading("3.2 GPU 耗时构成(TensorRT 11.3,batch 8,ms)", level=2)
make_table(doc, ["部分", "FP16", "FP8 默认", "FP8 优化后"], [
    ["线性层 GEMM", "1.52", "1.05", "1.06"],
    ["GELU", "0.00(已融合)", "0.20", "0.00(已融合)"],
    ["LayerNorm(含残差 / 移位 / 窗口重排)", "0.33", "0.33", "0.30"],
    ["注意力(fused MHA)", "0.31", "0.29", "0.29"],
    ["数据重排", "0.20", "0.17", "0.16"],
    ["分类头 GEMM(21841 类)", "0.13", "0.14", "0.07"],
    ["Cast / 其它", "0.04", "0.05", "0.03"],
    ["合计", "2.53", "2.23", "1.91"],
], widths=[2.6, 1.3, 1.3, 1.3], highlight_row=1)
note("FP8 优化后 = GELU tanh 近似 + 分类头量化(见 3.3)。线性层 FP8 加速 1.45×;LayerNorm、注意力、数据重排、分类头"
     "合计占 FP16 的 38%,默认 FP8 不加速。逐层 profile 含插桩开销,仅作相对比较。")

doc.add_heading("3.3 FP8 优化变体(TensorRT 11.3,相对 FP16)", level=2)
make_table(doc, ["变体", "改动", "batch 1", "batch 8", "batch 32", "top-1 一致率"], [
    ["FP8 默认", "ModelOpt 默认", "1.30×", "1.20×", "1.27×", "88.0%"],
    ["注意力不量化", "--disable_mha_qdq", "1.29×", "1.19×", "1.22×", "90.8%"],
    ["torch 量化路径", "modelopt.torch 量化后导出", "1.17×", "0.86×", "0.74×", "89.3%"],
    ["分类头量化", "按部署 batch 判断 GEMV", "1.41×", "1.25×", "1.28×", "87.8%"],
    ["GELU tanh", "导出时 GELU 用 tanh 近似", "1.34×", "1.31×", "1.54×", "88.0%"],
    ["GELU tanh + 分类头量化", "两项叠加", "1.45×", "1.37×", "1.56×", "88.0%"],
], widths=[1.45, 1.75, 0.75, 0.75, 0.75, 0.95], highlight_row=5)
note("基线均为原模型(erf GELU)的 FP16。GELU tanh 与原写法在 FP32 下 logits 余弦 ≥ 0.99999,对精度可忽略;"
     "各 FP8 变体与 FP16 的 logits 余弦均约 0.989,一致率相近,改动未引入额外精度损失。")

# ── 四、原因分析 ──
doc.add_heading("四、原因分析", level=1)
make_table(doc, ["因素", "证据", "影响"], [
    ["计算构成", "线性层 GEMM 占 FP16 GPU 时间 60%;其余 38% FP8 不加速", "batch 8 下即使 GEMM 加速 2 倍,上限约 1.43×"],
    ["GELU 融合断开", "FP16 中 GELU 融合进 GEMM;FP8 中成为独立 kernel(0.20 ms)", "抵消 GEMM 节省的 42%"],
    ["分类头被跳过", "ModelOpt 校准按 batch 1 推理,把 21841 类分类头误判为 GEMV", "0.14 ms 未加速"],
    ["注意力尺寸小", "head 维度 32、窗口 49 token,低于 TRT FP8 注意力的收益门槛(≥ 128)", "量化注意力仅快约 2%"],
    ["torch 量化路径", "导出图使 TRT 11.3 无法融合注意力;TRT 10.13 构建失败", "比 FP16 更慢"],
], widths=[1.3, 3.6, 1.8])
note("GELU 的融合与写法相关:TRT 11.3 中 FP16 只能融合 erf 写法,FP8 只能融合 tanh 写法。"
     "因此用 tanh 写法时 FP16 反而变慢(batch 8 −7%),但 FP8 获得完整融合。")

# ── 五、建议 ──
doc.add_heading("五、建议", level=1)
make_table(doc, ["建议", "依据"], [
    ["导出 ONNX 时 GELU 用 tanh 近似(PyTorch:nn.GELU(approximate=\"tanh\")),再做 FP8 量化", "batch 8:1.20× → 1.31×;精度不变"],
    ["ModelOpt 量化加 --calibration_shapes=input:<部署 batch>x3x224x224", "分类头一并量化,batch 1 / 8 再提升 8% / 4%"],
    ["继续使用 ModelOpt ONNX PTQ,不用 torch 量化路径", "torch 路径更慢,且 TRT 10 构建失败"],
    ["不必为注意力 FP8 投入", "全部留 FP16 仅慢约 2%"],
    ["在线服务提高 batch(动态批处理)", "两项优化后 batch 32 达 1.56×"],
    ["用客户自有模型与验证集确认精度", "本测试只比较与 FP16 的输出一致性"],
], widths=[4.2, 2.5])

# ── 六、说明 ──
doc.add_heading("六、说明", level=1)
bullet(doc, "测试硬件为 Blackwell;客户 L4 为 Ada 架构,算子融合行为可能不同,GELU 写法与分类头量化的收益需在 L4 上复核。", size=9.5)
bullet(doc, "吞吐以 trtexec 整体计时为准;逐层 profile 仅用于耗时构成的相对比较。", size=9.5)
bullet(doc, "精度为与 FP16 引擎输出的一致性;22K 类中近义类多,top-1 一致率只作变体间相对比较,不代表绝对准确率。", size=9.5)
bullet(doc, "客户若使用 1000 类或自定义分类头,分类头占比更小,「分类头量化」的收益相应减小。", size=9.5)
bullet(doc, "各变体与同一轮运行的 FP16 基线比较;GELU tanh 两行的基线取第二轮原模型 FP16。同一 FP16 基线两次测量相差约 0.5%。", size=9.5)

out = Path(__file__).with_name("report_swin.docx")
doc.save(out)
print(f"已生成 {out}")
