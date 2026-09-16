import cv2
from ultralytics import YOLO
model=YOLO("runs/detect/train-2/weights/best.pt")
video=cv2.VideoCapture("car_video.mp4")
while True:
    success, frame=video.read()
    if not success:
        break
    result=model(frame)
    boxes=result[0].boxes
    print("Number of plates detected :",len(boxes))
    for i,box in enumerate(boxes):
        x1,y1,x2,y2=box.xyxy[0].tolist()
        x1,y1,x2,y2=map(int,[x1,y1,x2,y2])
        print(
            "plate",i,
            "cordinated",
            x1,y1,x2,y2
        )

video.release()