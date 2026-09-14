# We will be testing the model using images from the images folder
#we will save the detected and cropped plates in the results folder
#after that we will run the OCR we will use  the easyocr
import os
import cv2
import csv
from ultralytics import YOLO
import easyocr
model=YOLO("runs/detect/train-2/weights/best.pt")
reader=easyocr.Reader(["en"])
image_folder="images"
results_folder = "results" 
annotated_folder = os.path.join(results_folder, "annotated")
os.makedirs(results_folder, exist_ok=True)
os.makedirs(annotated_folder, exist_ok=True)
image_files=os.listdir(image_folder)
# we are going to save the results into csv file 
csv_path=os.path.join(results_folder,"plate_results.csv")
csv_file=open(csv_path,"w",newline="")
csv_writer=csv.writer(csv_file)
csv_writer.writerow([
    "Image",
    "Plate",
    "YOLO Confidence",
    "OCR Confidence"
])
for filename in image_files:
    image_path=os.path.join(image_folder,filename)
    print("Processing ",filename)
    image=cv2.imread(image_path)
    if image is None:
        print("Image could not be found ",filename)
        continue
    result=model(image)
    boxes=result[0].boxes
    for i,box in enumerate(boxes):
        x1,y1,x2,y2=box.xyxy[0].tolist()
        x1,y1,x2,y2=map(int,[x1,y1,x2,y2])
        detection_confidence=float(box.conf[0])
        plate_crop=image[y1:y2,x1:x2]
        ocr_results=reader.readtext(plate_crop)
        for bbox,text,ocr_confidence in ocr_results:
            print("Lincense plate :",text)
            print("YOLO confidence :",detection_confidence)
            print("OCR confidence :",ocr_confidence)
            #now fill the csv rows
            csv_writer.writerow([
                filename,
                text,
                detection_confidence,
                ocr_confidence
            ])
        #Draw a rectangle around the plate where the plate has been detected by YOLO
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
        if ocr_results:
            cv2.putText(image,text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        output_path = os.path.join(results_folder,f"{os.path.splitext(filename)[0]}_plate_{i}.jpg")
        cv2.imwrite(output_path, plate_crop)
        print("Saved crop:", output_path)

    annotated_path = os.path.join(annotated_folder,f"{os.path.splitext(filename)[0]}_result.jpg")
    cv2.imwrite(annotated_path, image)

    print("Annotated image saved:", annotated_path)

    print("-" * 50)
csv_file.close()
print("CSV Saved :",csv_path)
