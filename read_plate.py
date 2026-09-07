import cv2
import easyocr

image=cv2.imread("plate_0.jpg")
if image is None:
    print("Image could not be found")
    exit()
reader=easyocr.Reader(["en"])
results=reader.readtext(image)
for bbox, text, confidence in results:
    print("Number Plate ::", text)
    print("Confidence ::",confidence)