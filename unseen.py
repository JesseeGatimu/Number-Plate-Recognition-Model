from ultralytics import YOLO
import cv2
model=YOLO("runs/detect/train-2/weights/best.pt")
image=cv2.imread("two_cars.jpeg")
result=model(image)
annonated=result[0].plot()
cv2.imshow("Detected plate",annonated)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(result[0].boxes)