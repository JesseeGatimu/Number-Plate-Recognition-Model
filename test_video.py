import cv2
video=cv2.VideoCapture("car_video.mp4")
while True:
    success, frame=video.read()
    if not success:
        break
    print("Video read successfully .")
video.release()