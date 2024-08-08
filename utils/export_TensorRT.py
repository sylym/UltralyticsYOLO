import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

from ultralytics import YOLO

if __name__ == "__main__":
    # 加载YOLOv8模型
    model = YOLO("../models/yolov10m2.pt")

    # 将模型导出为TensorRT格式
    model.export(
        format="engine",
        half=True,
        batch=7
    )
