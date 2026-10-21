import sys
from pathlib import Path

# 二期脚本在 quant/ 下 (容器里以仓库根目录为当前目录运行 python quant/xxx.py, 同目录互相 import)
sys.path.insert(0, str(Path(__file__).parent.parent / "quant"))
