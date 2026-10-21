"""被测模型: 产物文件名、输入张量、校准集、Triton 配置模板、可用的量化标签。各脚本按 --model 取, 默认 yolov8s。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str               # 产物前缀: artifacts/<name>.onnx, Triton 模型 <name>_<变体>
    input_name: str
    chw: tuple
    calib: str              # ModelOpt 校准集 (N,C,H,W) float32 .npy, 预处理与评测一致
    triton_template: str    # config.pbtxt 模板, 只改模型名与 I/O 数据类型
    quant_tags: tuple       # 该模型可用的量化标签 (见 variants.QUANT_TAGS)

    @property
    def source_onnx(self):
        return f"artifacts/{self.name}.onnx"

    @property
    def autocast_calib(self):
        """AutoCast 判断数值范围用的少量样本 (校准集前 16 张), .npz 键 = 输入名。"""
        return f"datasets/autocast_calib_{self.name}.npz"

    @property
    def fp16_onnx(self):
        """ModelOpt AutoCast 生成的 FP16 ONNX (I/O 保持 FP32), TRT 11 的 FP16 基线。"""
        return f"artifacts/{self.name}.fp16.onnx"

    def shape(self, batch):
        """trtexec --shapes / --optShapes 的写法, 如 images:8x3x640x640。"""
        return f"{self.input_name}:{batch}x" + "x".join(map(str, self.chw))


MODELS = {m.name: m for m in [
    # 一期 / 二期: 检测, 卷积为主
    Model("yolov8s", "images", (3, 640, 640), "datasets/calib_500.npy",
          "model_repository/yolov8s/config.pbtxt", ("nohead", "conv")),
    # 三期: 分类, Transformer (timm swin_tiny_patch4_window7_224.ms_in22k, 21841 类)
    Model("swin_t", "input", (3, 224, 224), "datasets/calib_swin_t_500.npy",
          "quant/triton_templates/swin_t.pbtxt", ("nomha", "torch")),
]}


def get(name):
    if name not in MODELS:
        raise ValueError(f"未知模型 {name!r} (可选 {' / '.join(MODELS)})")
    return MODELS[name]
