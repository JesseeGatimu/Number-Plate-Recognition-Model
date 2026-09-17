import cv2
from ultralytics import YOLO
import os
model=YOLO("runs/detect/train-2/weights/best.pt")
video=cv2.VideoCapture("car_video.mp4")
os.makedirs("video_plates",exist_ok=True)
frame_number=0
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_interval = total_frames // 5

frame_number = 0

while True:
    success, frame=video.read()
    if not success:
        break
    frame_number+=1

    if frame_number % frame_interval != 0:
        continue
    result=model(frame)
    boxes=result[0].boxes
    print("Number of plates detected :",len(boxes))
    for i,box in enumerate(boxes):
        x1,y1,x2,y2=box.xyxy[0].tolist()
        x1,y1,x2,y2=map(int,[x1,y1,x2,y2])
        plate_crop=frame[y1:y2,x1:x2]
        print(
            "plate",i,
            "cordinated",
            x1,y1,x2,y2
        )
        crop_path = os.path.join(
            "video_plates",
            f"plate_{frame_number}_{i}.jpg"
        )

        cv2.imwrite(crop_path, plate_crop)

video.release()