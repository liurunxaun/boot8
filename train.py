from ultralytics import YOLO
import os

current_path = os.path.dirname(__file__)

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(task="detect", data=f'{current_path}/mydata/VOC12BootTT.yaml', epochs=1, imgsz=640)