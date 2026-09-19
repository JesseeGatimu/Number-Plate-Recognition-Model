import cv2
import csv
import os
import easyocr
from ultralytics import YOLO

# Load YOLO model
model = YOLO("runs/detect/train-2/weights/best.pt")

# Load EasyOCR
reader = easyocr.Reader(["en"])

# Open video
video = cv2.VideoCapture("car_video.mp4")

if not video.isOpened():
    print("Could not open video")
    exit()

# Get video information
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Width:", width)
print("Height:", height)
print("Total frames:", total_frames)

# Create output folders
os.makedirs("video_results", exist_ok=True)

# Create output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output_video = cv2.VideoWriter(
    "video_results/output.mp4",
    fourcc,
    fps,
    (width, height)
)

# Create CSV file
csv_path = "video_results/video_plate_results.csv"

csv_file = open(
    csv_path,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Frame",
    "Plate",
    "YOLO Confidence",
    "OCR Confidence"
])

# Process video
frame_number = 0

while True:

    # Read frame
    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    # YOLO detection
    result = model(frame)

    boxes = result[0].boxes

    print(
        "Frame:",
        frame_number,
        "| Plates detected:",
        len(boxes)
    )

    # Process each detected plate
    for i, box in enumerate(boxes):

        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        x1, y1, x2, y2 = map(
            int,
            [x1, y1, x2, y2]
        )

        # YOLO confidence
        detection_confidence = float(box.conf[0])

        # Crop plate
        plate_crop = frame[y1:y2, x1:x2]

        # Run OCR
        ocr_results = reader.readtext(plate_crop)

        # Draw YOLO box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Process OCR result
        if ocr_results:

            for bbox, text, ocr_confidence in ocr_results:

                print(
                    "License plate:",
                    text
                )

                print(
                    "YOLO confidence:",
                    detection_confidence
                )

                print(
                    "OCR confidence:",
                    ocr_confidence
                )

                # Save result to CSV
                csv_writer.writerow([
                    frame_number,
                    text,
                    detection_confidence,
                    ocr_confidence
                ])

                # Draw plate text
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        else:

            print(
                "OCR could not read the plate"
            )

    # Write frame to output video
    output_video.write(frame)

# Release everything
video.release()
output_video.release()
csv_file.close()

print("VIDEO PROCESSING COMPLETE")
print("Output video:", "video_results/output.mp4")
print("CSV results:", csv_path)