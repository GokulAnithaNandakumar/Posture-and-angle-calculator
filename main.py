import math
import cv2
import numpy as np
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from pipline import *
import time

# Setup Pose function for video using MediaPipe's Holistic model.
mp_holistic = mp.solutions.holistic
pose_video = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.7, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
# camera_video.set(3, 1280)
# camera_video.set(4, 960)

# Initialize a resizable window.
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
# Main Video Processing Loop
while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

    t1 = time.time()
    frame, landmarks = detectPose(frame, pose_video, display=False)

    if landmarks:
        frame, _ = classifyPose(landmarks, frame, display=False)

    t2 = time.time() - t1
    cv2.putText(frame, "{:.0f} ms".format(t2 * 1000), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (203, 52, 247), 1)
    cv2.imshow('Pose Classification', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera_video.release()
cv2.destroyAllWindows()