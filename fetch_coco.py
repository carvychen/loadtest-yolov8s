"""下载 COCO val2017 图片 (~780MB) + 标注 (~250MB) 到 datasets/coco。一次性, 由 run_quant.sh 调用。"""
import os
import urllib.request
import zipfile

from quant_common import ANN, COCO_DIR, IMG_DIR

URLS = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
DONE = {"val2017.zip": IMG_DIR, "annotations_trainval2017.zip": ANN}

os.makedirs(COCO_DIR, exist_ok=True)
for name, url in URLS.items():
    if os.path.exists(DONE[name]):
        print(f"已存在 {DONE[name]}, 跳过")
        continue
    zpath = os.path.join(COCO_DIR, name)
    if not os.path.exists(zpath):
        print(f"下载 {url} ...")
        urllib.request.urlretrieve(url, zpath + ".part")
        os.replace(zpath + ".part", zpath)
    print(f"解压 {name} ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(COCO_DIR)
    os.remove(zpath)

n = len(os.listdir(IMG_DIR))
assert n == 5000, f"val2017 应有 5000 张, 实际 {n}"
print(f"COCO val2017 就绪: {n} 张 + {ANN}")
