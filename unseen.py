#Testing the model with some unseen images to see how it performs
from ultralytics import YOLO
import cv2
model=YOLO(
    "runs/detect/train-2/weights/best.pt"
)
img=cv2.imread("gari.jpeg")
results=model(img)
annonated_image=results[0].plot()
print(results[0].boxes)
cv2.imshow("Detected plate",annonated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
