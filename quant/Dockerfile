# 量化 + 精度评测镜像 (run_quant.sh 以 quant/ 为 build context, 本镜像不 COPY 任何文件): NVIDIA ModelOpt ONNX PTQ + Triton gRPC 客户端 + pycocotools。
# 不需要 GPU: 校准走 ONNX Runtime CPU EP (只算激活范围, 与目标卡无关; 量化后的 Q/DQ ONNX 跨卡通用),
# 评测通过 gRPC 调 Triton 上真实的 TRT 引擎。
# ModelOpt 硬依赖 torch, 先装 CPU 版避免拉 2GB+ 的 CUDA torch。
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "nvidia-modelopt[onnx]==0.47.0" opencv-python-headless pycocotools \
    "tritonclient[grpc]" pandas
WORKDIR /work
