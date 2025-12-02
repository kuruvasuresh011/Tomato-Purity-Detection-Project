from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")    # use yolov8s.pt for higher accuracy

# Train the model
results = model.train(
    data="data.yaml",         # path to your yaml file
    epochs=50,                # increase if needed (recommended 50–100)
    imgsz=640,
    batch=8,
)
##yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
