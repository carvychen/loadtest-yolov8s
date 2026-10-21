import sys
from pathlib import Path

# 被测脚本都在仓库根目录 (容器里以 /work 为当前目录直接运行), 测试从任意目录跑都能 import
sys.path.insert(0, str(Path(__file__).parent.parent))
