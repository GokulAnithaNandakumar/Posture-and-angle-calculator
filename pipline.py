import math
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import time

# Time
time_ = 0

# Initializing mediapipe pose class.
# mp_pose = mp.solutions.pose
mp_pose = mp.solutions.holistic

mp_drawing_styles = mp.solutions.drawing_styles
# mp_holistic = mp.solutions.holistic

# Setting up the Pose function.
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

def drawTextOnImage(image, texts):
    for i, (text, value) in enumerate(texts):
        cv2.putText(image, f'{text}: {value}', (10, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

def updatePreviousAngles():
    global left_elbow_angle_previous, right_elbow_angle_previous, left_shoulder_angle_previous, right_shoulder_angle_previous, left_wrist_angle_previous, right_wrist_angle_previous
    left_elbow_angle_previous = left_elbow_angle
    right_elbow_angle_previous = right_elbow_angle
    left_shoulder_angle_previous = left_shoulder_angle
    right_shoulder_angle_previous = right_shoulder_angle
    left_wrist_angle_previous = left_wrist_angle
    right_wrist_angle_previous = right_wrist_angle


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, left_hand, right_hand])

def extract_keypoints_Pose(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        return np.concatenate([pose])

def extract_keypoints_Pose_(results):
        pose = []
        for i in range(len(results.pose_landmarks.landmark)):
            sample = [results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y]
            pose.append(sample)
        pose = np.array(pose)
        pose = pose.reshape((1,*pose.shape))
        return pose


def detectPose(image, pose, display=True):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)

    height, width, _ = image.shape
    landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.left_hand_landmarks,
                                  connections=mp_pose.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.right_hand_landmarks,
                                  connections=mp_pose.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.face_landmarks,
                                  connections=mp_pose.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))

        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))
        else:
            landmarks.extend([(0, 0, 0)] * 21)

        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))
        else:
            landmarks.extend([(0, 0, 0)] * 21)

    if display:
        plt.figure(figsize=[22, 22])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off')
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off')
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        return output_image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle

def calculateDistance(landmark1, landmark2):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2

    # Calculate the Distance between the two points
    dis = math.sqrt( ((x2 - x1)**2)+((y2 - y1))**2)


    # Return the calculated Distance.
    return dis

left_elbow_angle = 0;
right_elbow_angle = 0;
left_shoulder_angle = 0;
right_shoulder_angle = 0;
left_wrist_angle = 0;
right_wrist_angle = 0;

left_elbow_angle_previous = 0;
right_elbow_angle_previous = 0;
left_shoulder_angle_previous = 0;
right_shoulder_angle_previous = 0;
left_wrist_angle_previous = 0;
right_wrist_angle_previous = 0;

left_elbow_angle_diff = 0;
right_elbow_angle_diff = 0;
left_shoulder_angle_diff = 0;
right_shoulder_angle_diff = 0;
left_wrist_angle_diff = 0;
right_wrist_angle_diff = 0;

Angle_previous = []

def updateAngle_previous():
    # Angle_previous = []
    left_elbow_angle_previous = left_elbow_angle;
    right_elbow_angle_previous = right_elbow_angle;
    left_shoulder_angle_previous = left_shoulder_angle;
    right_shoulder_angle_previous = right_shoulder_angle;
    left_wrist_angle_previous = left_wrist_angle;
    right_wrist_angle_previous = right_wrist_angle;
    Angle_previous.append([left_elbow_angle_previous,right_elbow_angle_previous,left_shoulder_angle_previous,right_shoulder_angle_previous,left_wrist_angle_previous,right_wrist_angle_previous])


Angle_diff = []

def updateAngle_diff():
    # Angle_diff = []
    left_elbow_angle_diff = left_elbow_angle - left_elbow_angle_previous;
    right_elbow_angle_diff = right_elbow_angle - right_elbow_angle_previous;
    left_shoulder_angle_diff = left_shoulder_angle - left_shoulder_angle_previous;
    right_shoulder_angle_diff = right_shoulder_angle - right_shoulder_angle_previous;
    left_wrist_angle_diff = left_wrist_angle - left_wrist_angle_previous;
    right_wrist_angle_diff = right_wrist_angle - right_wrist_angle_previous;
    Angle_diff.append([left_elbow_angle_diff,right_elbow_angle_diff,left_shoulder_angle_diff,right_shoulder_angle_diff,left_wrist_angle_diff,right_wrist_angle_diff])


def classifyPose(landmarks, output_image, display=False):
    global time_, Angle_previous, Angle_diff
    global left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle, left_wrist_angle, right_wrist_angle
    global left_elbow_angle_previous, right_elbow_angle_previous, left_shoulder_angle_previous, right_shoulder_angle_previous, left_wrist_angle_previous, right_wrist_angle_previous
    global left_elbow_angle_diff, right_elbow_angle_diff, left_shoulder_angle_diff, right_shoulder_angle_diff, left_wrist_angle_diff, right_wrist_angle_diff

    label = 'Unknown Pose'
    color = (0, 0, 255)

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    LEFT_HAND, RIGHT_HAND = 33, 33 + 21
    left_wrist_angle = calculateAngle(landmarks[mp_pose.HandLandmark.THUMB_TIP.value + LEFT_HAND],
                                      landmarks[mp_pose.HandLandmark.WRIST.value + LEFT_HAND],
                                      landmarks[mp_pose.HandLandmark.PINKY_TIP.value + LEFT_HAND])
    right_wrist_angle = calculateAngle(landmarks[mp_pose.HandLandmark.THUMB_TIP.value + RIGHT_HAND],
                                       landmarks[mp_pose.HandLandmark.WRIST.value + RIGHT_HAND],
                                       landmarks[mp_pose.HandLandmark.PINKY_TIP.value + RIGHT_HAND])

    left_thumb_index_distance = calculateDistance(landmarks[mp_pose.HandLandmark.THUMB_TIP.value + LEFT_HAND],
                                                  landmarks[mp_pose.HandLandmark.INDEX_FINGER_TIP.value + LEFT_HAND])
    right_thumb_index_distance = calculateDistance(landmarks[mp_pose.HandLandmark.THUMB_TIP.value + RIGHT_HAND],
                                                   landmarks[mp_pose.HandLandmark.INDEX_FINGER_TIP.value + RIGHT_HAND])
    left_thumb_pinky_distance = calculateDistance(landmarks[mp_pose.HandLandmark.THUMB_TIP.value + LEFT_HAND],
                                                  landmarks[mp_pose.HandLandmark.PINKY_TIP.value + LEFT_HAND])
    right_thumb_pinky_distance = calculateDistance(landmarks[mp_pose.HandLandmark.THUMB_TIP.value + RIGHT_HAND],
                                                   landmarks[mp_pose.HandLandmark.PINKY_TIP.value + RIGHT_HAND])

    drawTextOnImage(output_image, [
        ("L_elbow_angle", left_elbow_angle),
        ("R_elbow_angle", right_elbow_angle),
        ("L_shoulder_angle", left_shoulder_angle),
        ("R_shoulder_angle", right_shoulder_angle),
        ("L_wrist_angle", left_wrist_angle),
        ("R_wrist_angle", right_wrist_angle),
        ("L_knee_angle", left_knee_angle),
        ("R_knee_angle", right_knee_angle),
        ("L_thumb_index_distance", left_thumb_index_distance),
        ("R_thumb_index_distance", right_thumb_index_distance),
        ("L_thumb_pinky_distance", left_thumb_pinky_distance),
        ("R_thumb_pinky_distance", right_thumb_pinky_distance),
    ])

    if right_elbow_angle > 165 and right_elbow_angle < 195 and left_elbow_angle > 165 and left_elbow_angle < 195:
        if right_shoulder_angle > 80 and right_shoulder_angle < 110 and left_shoulder_angle > 80 and left_shoulder_angle < 110:
            if right_knee_angle > 165 and right_knee_angle < 195 or left_knee_angle > 165 and left_knee_angle < 195:
                label = 'T Pose'

    updatePreviousAngles()

    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)

    if display:
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')
    else:
        return output_image, label
