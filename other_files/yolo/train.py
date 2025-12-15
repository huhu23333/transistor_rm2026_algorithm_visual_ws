from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
#model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="已标注数据集.yaml", epochs=100, imgsz=640, 
    hsv_h=0.1,
    hsv_s=0.7,
    hsv_v=0.7,
    degrees=10.0,
    translate=0.2,
    scale=0.5,
    shear=10.0,
    perspective=0.001,
    fliplr=0.5,
    mosaic=1.0,
    auto_augment=None) 
