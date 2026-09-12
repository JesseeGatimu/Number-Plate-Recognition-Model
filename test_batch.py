import os
from ultralytics import YOLO
model=YOLO("runs/detect/train-2/weights/best.pt")
image_folder="images"
image_files=os.listdir(image_folder)
for filename in image_files:
    image_path=os.path.join(image_folder,filename)
    print("Processing ",filename)
    result=model(image_path)
