from ultralytics import YOLO
import easyocr
import cv2
import subprocess

model=YOLO("runs/detect/train-2/weights/best.pt")
reader=easyocr.Reader(["en"])
image=cv2.imread("prado.jpg")

if image is None:
    print("Could not load image !")
    exit()
result=model(image)
boxes=result[0].boxes

for i, box in enumerate(boxes):
    x1,y1,x2,y2=box.xyxy[0].tolist()
    x1,y1,x2,y2=map(int,[x1,y1,x2,y2])
    detection_confidence=float(box.conf[0])
    #now let us crop the image 
    plate_crop=image[y1:y2,x1:x2]
    #read the number plate after cropping the image 
    ocr_results=reader.readtext(plate_crop)

    if not ocr_results:
        print("No number plate read !!")
        continue
    for bbox,text,ocr_confidence in ocr_results:
        print("------------------------")
        print("Licence Plate : ",text)
        print("YOLO Confidence :",detection_confidence)
        print("OCR confidence :",ocr_confidence)
        print("------------------------")

    #let us draw a rectangle around the number plate
    cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
    #write the number plate on top 
    cv2.putText(image,text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

success=cv2.imwrite("Final_result.jpg",image)

if success:
    print("Image saved successfully")
else:
    print("Image could not be saved")
subprocess.run(["xdg-open","Final_result.jpg"])
