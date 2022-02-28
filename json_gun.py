import torch
import json
import cv2
from time import time

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/content/drive/MyDrive/Brain Station 23/RnD Lab/Augustus/Violence_Gun_detection/yolov5/runs/train/test1/weights/best.pt')   # or yolov5m, yolov5l, yolov5x, custom

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/content/drive/MyDrive/Brain Station 23/RnD Lab/Augustus/Violence_Gun_detection/yolov5/runs/train/test1/weights/best.pt')   # or yolov5m, yolov5l, yolov5x, custom

results = model(img)
results.pandas().xyxy[0]

result = results.pandas().xyxy[0].to_json(orient = "records")
parsed = json.loads(result)
json.dumps(parsed, indent=4)