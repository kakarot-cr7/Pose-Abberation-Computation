import math
import cv2
import numpy as np
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from pipeline2 import *
import time
mp_drawing = mp.solutions.drawing_utils

# Setup Pose function for video.
# pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=1)
pose_video = mp_pose.Holistic(static_image_mode=True, min_detection_confidence=0.7, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
#camera_video = cv2.VideoCapture(0)
image = cv2.imread('C:/Users\RAJARSHI SAHA/Downloads/gettyimages-1169184617-2048x2048.jpg',cv2.IMREAD_COLOR)
# camera_video.set(3,1280)
# camera_video.set(4,960)


# Initialize a resizable window.
# cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
with pose_video as pose:
    
        
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = image
        
        # Get the width and height of the frame
        
        # Get the width and height of the frame
        frame_height, frame_width, _ =  frame.shape
        
        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        # print(frame.shape)
        t1 = time.time()
        # Perform Pose landmark detection.
        frame, landmarks = detectPose(frame, pose_video, display=True)
        

        results = pose.process(frame)
        # Check if the landmarks are detected.
        if landmarks:
            
            # Perform the Pose Classification.
            frame, _ = classifyPose(landmarks, frame, display=True)
        
        t2 = time.time() - t1
        cv2.putText(frame, "{:.0f} ms".format(
                t2*1000), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (203, 52, 247), 1)
        # Display the frame.
        cv2.imshow('Pose Classification', frame)
        
        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(0) & 0xFF
        # Check if 'ESC' is pressed.
            # Break the loop. 
mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,)
print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
# Release the VideoCapture object and close the windows.
image.release()
cv2.destroyAllWindows()