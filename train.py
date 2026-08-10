from ultralytics import YOLO
model=YOLO("yolov8n")
model.train(
    data="License-Plate-Data/data.yaml",
    epochs=30,
    batch=2,
    imgsz=640,
    device="cpu"
)