"""精度变体: 变体名 -> 量化参数 / ONNX 文件 / 引擎构建参数 / Triton 模型名。二期各脚本只从这里取, 不自己解析变体名。
每个变体属于一个模型 (models.py, 默认 yolov8s): 文件名、校准集、可用的量化标签都随模型。"""
from dataclasses import dataclass

import models
from models import Model
# FP8 用 max (E4M3 动态范围大, NVIDIA 推荐); INT8 用 entropy (TensorRT 经典 CNN 口径)
CALIBRATION = {"fp8": "max", "int8": "entropy"}
# 量化标签: 改变 Q/DQ ONNX (值 = 追加的 ModelOpt ONNX PTQ 参数, 或按模型生成参数的函数)。字典顺序 = ONNX 文件名里的标签顺序
QUANT_TAGS = {
    "nohead": ["--nodes_to_exclude", r"/model\.22/.*"],   # YOLO: 检测头 model.22 不量化
    "conv": ["--op_types_to_quantize", "Conv"],           # YOLO: 只量化 Conv (不量化残差 Add)
    "nomha": ["--disable_mha_qdq"],                       # 注意力的两个 BMM (QK^T / ·V) 不加 Q/DQ, 留 FP16
    # 分类头也量化: ModelOpt 默认按 batch 1 推理找 GEMV (m 或 n 为 1), 把分类头 Gemm 当矩阵向量乘排除;
    # 改按部署的 batch 8 做形状推断, 分类头就是普通 GEMM
    "head": lambda m: [f"--calibration_shapes={m.shape(8)}"],
    "torch": None,                                        # 不走 ONNX PTQ: modelopt.torch 量化后导出 (swin_torch_quant.py)
}
# 构建标签: 同一 ONNX, 只改 trtexec 构建参数
BUILD_TAGS = {
    "opt5": ["--builderOptimizationLevel=5"],             # 默认 3, 搜更多 tactic / 融合
}


@dataclass(frozen=True)
class Variant:
    name: str
    precision: str
    quant_tags: tuple
    build_tags: tuple
    model: Model

    @classmethod
    def parse(cls, name, model="yolov8s"):
        """变体名 = <精度>[-标签...], 标签顺序不限。拼错 / 不支持的组合直接报错, 不悄悄退化成别的变体。"""
        m = models.get(model)
        precision, *tags = name.split("-")
        if precision not in ("fp16", *CALIBRATION):
            raise ValueError(f"变体 {name!r}: 未知精度 {precision!r} (可选 fp16 / fp8 / int8)")
        for t in tags:
            if t not in QUANT_TAGS and t not in BUILD_TAGS:
                raise ValueError(f"变体 {name!r}: 未知标签 {t!r} (量化标签 {'/'.join(QUANT_TAGS)}, "
                                 f"构建标签 {'/'.join(BUILD_TAGS)})")
        if len(set(tags)) != len(tags):
            raise ValueError(f"变体 {name!r}: 标签重复")
        if precision == "fp16" and any(t in QUANT_TAGS for t in tags):
            raise ValueError(f"变体 {name!r}: fp16 不量化, 不能带量化标签")
        for t in tags:
            if t in QUANT_TAGS and t not in m.quant_tags:
                raise ValueError(f"变体 {name!r}: 标签 {t!r} 不适用于模型 {m.name} (可用 {'/'.join(m.quant_tags)})")
        if "torch" in tags and (precision != "fp8" or len([t for t in tags if t in QUANT_TAGS]) > 1):
            raise ValueError(f"变体 {name!r}: torch 路径只支持 fp8, 且不能叠加其它量化标签 (注意力 Q/DQ 由它自己处理)")
        return cls(name, precision, tuple(t for t in QUANT_TAGS if t in tags),
                   tuple(t for t in BUILD_TAGS if t in tags), m)

    @property
    def triton_name(self):
        return f"{self.model.name}_{self.name}"

    @property
    def quant_name(self):
        """量化模型名 = 精度 + 量化标签 (固定顺序), 同名变体共用一份 Q/DQ ONNX; fp16 不量化, 为 None。"""
        return None if self.precision == "fp16" else "-".join((self.precision,) + self.quant_tags)

    @property
    def model_dir(self):
        return f"artifacts/model_repository_q/{self.triton_name}"

    def onnx_path(self, trt_major=None):
        """构建引擎用的 ONNX。fp16 不量化: TRT 10 用原 FP32 ONNX + --fp16; TRT 11 已移除弱类型 (--fp16 等不存在),
        改用 ModelOpt AutoCast 生成的 FP16 ONNX (I/O 保持 FP32)。量化变体的 ONNX 与 TRT 版本无关, trt_major 可省略。"""
        if self.precision == "fp16":
            if trt_major is None:
                raise ValueError("fp16 的 ONNX 取决于 TRT 主版本")
            return self.model.source_onnx if trt_major < 11 else self.model.fp16_onnx
        return f"artifacts/{self.model.name}.{self.quant_name}.onnx"

    def quantize_cmd(self, calibration_method=None):
        """生成本变体 Q/DQ ONNX 的命令 (python 之后的参数); fp16 不量化, 返回 None。"""
        if self.precision == "fp16":
            return None
        if "torch" in self.quant_tags:
            return ["quant/swin_torch_quant.py", f"--calibration_data_path={self.model.calib}",
                    f"--output_path={self.onnx_path()}"]
        method = calibration_method or CALIBRATION[self.precision]
        args = ["-m", "modelopt.onnx.quantization", f"--onnx_path={self.model.source_onnx}",
                f"--quantize_mode={self.precision}", f"--calibration_data_path={self.model.calib}",
                f"--calibration_method={method}", "--calibration_eps", "cpu", f"--output_path={self.onnx_path()}"]
        for t in self.quant_tags:
            extra = QUANT_TAGS[t]
            args += extra(self.model) if callable(extra) else extra
        return args

    def build_flags(self, trt_major):
        """trtexec 构建引擎时的精度相关参数。TRT 11 强类型为默认, 不加 --stronglyTyped (已是空操作)。"""
        if trt_major < 11:
            flags = ["--fp16"] if self.precision == "fp16" else ["--stronglyTyped"]
        else:
            flags = []
        for t in self.build_tags:
            flags += BUILD_TAGS[t]
        return flags



def parse_all(names, model="yolov8s"):
    """命令行入口用: 解析一组变体名, 有一个不合法就打印原因退出。"""
    try:
        return [Variant.parse(n, model) for n in names]
    except ValueError as e:
        raise SystemExit(f"!! {e}")


if __name__ == "__main__":
    import argparse
    # 用法: python quant/variants.py check [--model M] <变体...>   (run_quant.sh 开跑前先校验, 拼错几秒内就报错)
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["check"])
    p.add_argument("--model", default="yolov8s")
    p.add_argument("variants", nargs="+")
    a = p.parse_args()
    parse_all(a.variants, a.model)
    print(f"变体 OK ({a.model}): " + " ".join(a.variants))
