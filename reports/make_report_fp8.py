"""生成二期 FP8 量化分析报告 (report_fp8.docx)。数据来源: results/quant/ (rtx6000-quarter-nc24[-trt11]-*)。
运行: python reports/make_report_fp8.py
"""
from pathlib import Path

from report_style import MUTED, bullet, make_table, new_doc, para

doc = new_doc(
    "YOLOv8s FP8 量化推理性能分析报告",
    "RTX PRO 6000 Blackwell(1/4 切片) · FP16 vs FP8 · TensorRT 10.13 / 11.3",
    "测试基础:Standard_NC24lds_xl_RTXPRO6000BSE_v6 · YOLOv8s 640×640 · NVIDIA ModelOpt PTQ · COCO val2017 · 2026-10")


def note(text):
    para(doc, text, size=9, color=MUTED, after=6)


# ── 结论摘要 ──
doc.add_heading("结论摘要", level=1)
bullet(doc, "TensorRT 11.3(最新)下,FP8 相比 FP16 GPU 吞吐提升 55%(batch 8)/ 50%(batch 32),mAP50-95 仅降 0.41 pt。")
bullet(doc, "同一 FP8 模型在 TensorRT 10.13 上仅提升 38% / 24%:TensorRT 版本是 FP8 能否超过 50% 的首要因素。")
bullet(doc, "batch 1 时两个版本均只提升 25%:小模型单张推理受 kernel 启动与访存限制,低精度收益有限。")
bullet(doc, "在线服务中 GPU 推理仅占请求延迟约 11%,FP8 的 GPU 收益被数据传输稀释,端到端几乎无提升。")
bullet(doc, "客户反馈的“FP8 提升不足 50%”与本测试一致,主要受 TensorRT 版本、batch 大小、端到端 I/O 三方面影响。")

# ── 一、测试环境 ──
doc.add_heading("一、测试环境", level=1)
make_table(doc, ["项目", "配置"], [
    ["Azure VM", "Standard_NC24lds_xl_RTXPRO6000BSE_v6(24 vCPU / 72 GiB)"],
    ["GPU", "RTX PRO 6000 Blackwell 1/4 MIG 切片 · 24 GB · 驱动 595.58"],
    ["推理栈", "Triton 25.08(TensorRT 10.13.2)/ Triton 26.09(TensorRT 11.3.0,当前最新)"],
    ["模型", "YOLOv8s 官方权重 · ONNX opset 17 · 640×640 · 动态 batch 1–32"],
    ["量化工具", "NVIDIA ModelOpt 0.47.0(ONNX PTQ,显式 Q/DQ)"],
    ["数据集", "COCO val2017:500 张校准 + 4500 张评测(不重叠)"],
], widths=[1.3, 5.4])

# ── 二、测试方法 ──
doc.add_heading("二、测试方法", level=1)
make_table(doc, ["环节", "方法"], [
    ["FP8 量化", "E4M3 · max 校准 · 量化 Conv / Add,其余算子 FP16"],
    ["FP16 基线", "TRT 10:FP32 ONNX + --fp16;TRT 11:AutoCast FP16 ONNX(TRT 11 已移除弱类型)。"
                  "两版本 FP16 吞吐一致(batch 8:1275 / 1293 img/s)"],
    ["引擎构建", "trtexec · 动态 batch(min 1 / opt 8 / max 32)· 同版本构建与运行"],
    ["性能", "trtexec GPU 纯计算吞吐(CUDA Graph)· batch 1 / 8 / 32 · 每点 15 s。含 PCIe 传输与纯计算相差约 3%,不单列"],
    ["精度", "Triton 加载实际引擎 · 4500 张 · mAP50-95(conf 0.001 / IoU 0.7)"],
    ["耗时归因", "trtexec 逐层 profile(batch 8)"],
    ["端到端", "Triton + perf_analyzer 逐请求延迟分解(并发 8,p99 30 ms)"],
    ["INT8 对照", "同流程(entropy 校准),用于区分“模型吃不到低精度”与“FP8 路径效率”"],
], widths=[1.3, 5.4])

# ── 三、测试结果 ──
doc.add_heading("三、测试结果", level=1)

doc.add_heading("3.1 FP8 vs FP16(GPU 纯计算吞吐 img/s,括号内为相对 FP16)", level=2)
make_table(doc, ["配置", "batch 1", "batch 8", "batch 32", "mAP50-95"], [
    ["TRT 10.13 · FP16", "858", "1,275", "1,208", "0.4515"],
    ["TRT 10.13 · FP8", "1,071(1.25×)", "1,757(1.38×)", "1,498(1.24×)", "0.4468(−0.47)"],
    ["TRT 11.3 · FP16", "877", "1,293", "1,128", "0.4513"],
    ["TRT 11.3 · FP8", "1,092(1.25×)", "2,002(1.55×)", "1,690(1.50×)", "0.4472(−0.41)"],
], widths=[1.6, 1.2, 1.2, 1.2, 1.5], highlight_row=3)
note("同一份 FP8 模型,仅升级 TensorRT,batch 8 吞吐 +14%(1,757 → 2,002 img/s)。")

doc.add_heading("3.2 FP8 优化变体(TensorRT 11.3)", level=2)
make_table(doc, ["变体", "改动", "batch 8", "mAP50-95", "结论"], [
    ["FP8(基准)", "默认 PTQ", "1.55×", "0.4472(−0.41)", "—"],
    ["仅量化 Conv", "残差 Add 不量化", "1.56×", "0.4474(−0.39)", "与基准相同"],
    ["优化等级 5", "builderOptimizationLevel 3→5", "1.55×", "0.4472(−0.41)", "无收益"],
    ["检测头 FP16", "model.22 不量化", "1.40×", "0.4491(−0.22)", "精度损失减半,速度约 −9%"],
], widths=[1.15, 1.85, 0.7, 1.25, 1.75])

doc.add_heading("3.3 GPU 耗时构成(batch 8,ms)", level=2)
make_table(doc, ["部分", "FP16 · 10.13", "FP8 · 10.13", "FP16 · 11.3", "FP8 · 11.3"], [
    ["卷积", "4.68", "3.13", "4.29", "2.66"],
    ["SiLU 激活", "0.64", "2.11", "1.46", "1.16"],
    ["格式转换", "1.12", "0.04", "0.00", "0.00"],
    ["Concat / Slice", "0.11", "0.46", "0.40", "0.15"],
    ["类型转换(Cast)", "0.00", "0.26", "0.16", "0.22"],
    ["检测头解码", "0.13", "0.18", "0.16", "0.16"],
    ["合计", "6.68", "6.17", "6.48", "4.35"],
], widths=[1.7, 1.25, 1.25, 1.25, 1.25], highlight_row=1)
note("FP8 卷积提速:10.13 为 1.50×,11.3 为 1.61×;10.13 中 SiLU 未与 FP8 卷积融合,开销反超 FP16。"
     "逐层 profile 含插桩开销,仅作相对比较。")

doc.add_heading("3.4 端到端延迟构成(Triton,并发 8,TRT 10.13,ms)", level=2)
make_table(doc, ["环节", "FP16", "FP8", "占比(FP16)"], [
    ["网络与服务端收发", "8.86", "8.87", "52%"],
    ["服务端输入 / 输出拷贝", "3.13", "3.51", "18%"],
    ["排队(动态批处理)", "2.40", "2.27", "14%"],
    ["GPU 推理", "1.80", "1.26", "11%"],
    ["客户端发送", "0.77", "0.77", "5%"],
    ["合计", "16.97", "16.68", "100%"],
], widths=[2.4, 1.2, 1.2, 1.5], highlight_row=3)
note("FP8 使 GPU 推理 −30%,请求总延迟仅 −1.7%。数据搬运相关环节约占 75%。")

# ── 四、原因分析 ──
doc.add_heading("四、原因分析", level=1)
make_table(doc, ["因素", "证据", "影响"], [
    ["TensorRT 版本", "同一 FP8 模型 batch 8:10.13 为 1.38×,11.3 为 1.55×", "首要因素"],
    ["batch 大小", "batch 1 两版本均 1.25×;batch 8 / 32 达 1.50× 以上", "低并发在线场景收益小"],
    ["SiLU 未融合", "11.3 FP8 中 SiLU 为独立 kernel,占 GPU 时间 27%", "GPU 侧剩余优化空间"],
    ["端到端 I/O", "GPU 推理仅占请求延迟 11%,数据搬运约 75%", "端到端收益被稀释"],
    ["模型本身", "同卡 INT8(TRT 10.13)达 1.87×", "排除“小模型吃不到低精度”"],
], widths=[1.3, 3.6, 1.8])

# ── 五、建议 ──
doc.add_heading("五、建议", level=1)
make_table(doc, ["建议", "依据"], [
    ["使用 TensorRT 11.3 及以上构建 FP8 引擎", "batch 8 加速 1.38× → 1.55×"],
    ["在线服务开启动态批处理,目标 batch ≥ 8", "batch 1 仅 1.25×,batch 8 达 1.55×"],
    ["精度敏感场景检测头保留 FP16", "mAP 损失 0.41 → 0.22 pt,速度约 −9%"],
    ["优化端到端 I/O:FP16 / uint8 输入、NMS 融入引擎、GPU 预处理", "数据搬运约占请求延迟 75%,是端到端提升的主要空间(详见第六节)"],
    ["跟进 TensorRT 后续版本的 FP8 Conv + SiLU 融合", "SiLU 占 FP8 GPU 时间 27%"],
], widths=[3.4, 3.3])

# ── 六、端到端 I/O 优化方向 ──
doc.add_heading("六、端到端 I/O 优化方向", level=1)
note("现状指本测试口径:Ultralytics 默认导出的模型输入为 FP32 归一化张量,客户端需先 /255 转 float32 再发送;"
     "每次请求传输输入 4.7 MB + 原始输出 2.7 MB,I/O 相关环节约 12.8 ms,占请求延迟 75%(见 3.4)。"
     "以下为优化方案与数据量估算,延迟收益需实测确认。")

doc.add_heading("6.1 优化方案", level=2)
make_table(doc, ["方案", "做法", "作用环节", "改动"], [
    ["uint8 输入", "模型前端加入 uint8→FP32 转换与 /255(重新导出 ONNX,随引擎在 GPU 执行);"
                   "Triton 输入改为 UINT8;客户端直接发送 letterbox 后的 uint8 图像",
     "网络收发、输入拷贝、客户端发送", "低"],
    ["NMS 融入引擎", "导出时加入 TensorRT EfficientNMS,引擎直接输出 ≤ 300 个检测框",
     "输出拷贝、网络收发、客户端 NMS", "低"],
    ["JPEG + GPU 解码", "客户端发送 JPEG;Triton DALI 后端在 GPU 上解码、缩放、letterbox",
     "网络收发、客户端预处理", "中"],
    ["共享内存", "客户端与 Triton 同机时,用 CUDA / 系统共享内存传张量,免去 gRPC 序列化",
     "网络收发、序列化", "低(仅同机)"],
    ["多实例", "instance_group 设 2 个实例,使数据拷贝与 GPU 计算重叠",
     "排队、输入 / 输出拷贝", "低"],
], widths=[1.15, 3.15, 1.65, 0.75])

doc.add_heading("6.2 单次请求数据量(640×640)", level=2)
make_table(doc, ["组合", "输入", "输出", "合计", "相对现状"], [
    ["现状:FP32 输入 + 原始输出", "4.7 MB", "2.7 MB", "7.4 MB", "—"],
    ["uint8 输入", "1.2 MB", "2.7 MB", "3.9 MB", "−47%"],
    ["uint8 输入 + NMS 融入引擎", "1.2 MB", "< 0.01 MB", "1.2 MB", "−84%"],
    ["JPEG + NMS 融入引擎", "约 0.1 MB", "< 0.01 MB", "约 0.1 MB", "约 −98%"],
], widths=[2.3, 1.0, 1.0, 1.0, 1.0], highlight_row=2)
note("建议先做 uint8 输入 + NMS 融入引擎:改动小,数据量 −84%。I/O 降下来后,FP8 的 GPU 收益才能在端到端体现。"
     "若线上已发送 uint8(由 Triton 前置预处理模型转换为 FP32),网络传输已优化,剩余收益在于将该转换移入 TensorRT 引擎在 GPU 执行。"
     "改为 uint8 输入后,FP8 量化的校准数据须同步改为 0–255 的 uint8,否则量化 scale 偏差 255 倍,精度失效。"
     "JPEG 大小随图像内容与压缩质量变化。")

# ── 七、说明 ──
doc.add_heading("七、说明", level=1)
bullet(doc, "测试硬件为 Blackwell;客户 L4 为 Ada 架构,FP8 kernel 不同,绝对加速比不可直接套用,TensorRT 版本、batch、I/O 三项影响因素同样需要考虑。", size=9.5)
bullet(doc, "吞吐以 trtexec 整体计时为准;逐层 profile 仅用于耗时构成的相对比较。", size=9.5)
bullet(doc, "mAP 采用固定 640×640 letterbox,绝对值略低于 Ultralytics 官方(rect 批推理),各精度同口径可比。", size=9.5)
bullet(doc, "端到端吞吐受压测工具计时粒度限制,不作为结论;端到端部分仅采用逐请求延迟分解。", size=9.5)

out = Path(__file__).with_name("report_fp8.docx")
doc.save(out)
print(f"已生成 {out}")
