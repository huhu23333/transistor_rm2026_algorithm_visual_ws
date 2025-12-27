from ultralytics import YOLO


model_path = "armor-oneclass-yolo11n-pose-best.pt"
model = YOLO(model_path)
model.export(format="openvino", int8=True, data="已标注数据集.yaml")
model.export(format="openvino")
