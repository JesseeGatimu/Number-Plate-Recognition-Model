from ultralytics import YOLO
import cv2
model=YOLO("runs/detect/train-2/weights/best.pt")
image=cv2.imread("prado.jpg")
if image is None:
    print("Image not found")
    exit()
results=model(image)
boxes=results[0].boxes
for i, box in enumerate(boxes):
    x1,y1,x2,y2=box.xyxy[0].tolist()
    x1,y1,x2,y2=map(int,[x1,y1,x2,y2])
    #crop the plate
    plate_crop=image[y1:y2, x1:x2]
    #save the cropped number plate
    cv2.imwrite(f"plate_{i}.jpg",plate_crop)
    print("Saved: plate_{i}.jpg")
    #Display the cropped plate
    cv2.imshow("Cropped Lincense plate",plate_crop)
    cv2.waitKey(0)
cv2.destroyAllWindow()

