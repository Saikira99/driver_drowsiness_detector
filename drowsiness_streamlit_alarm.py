import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
import winsound

# Streamlit page config
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("🚗 Driver Drowsiness Detection (Beep Version)")

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# EAR calculation
def eye_aspect_ratio(landmarks, eye_indices):
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Eye landmark indexes (from mediapipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR threshold & consecutive frame limit
EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 20
counter = 0

# Start webcam
cap = cv2.VideoCapture(0)

frame_placeholder = st.empty()
alert_placeholder = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                counter += 1
                if counter >= CLOSED_FRAMES:
                    alert_placeholder.error("🚨 Drowsiness Detected! Wake up!")
                    winsound.Beep(2000, 500)  # 2000 Hz for 0.5 sec
            else:
                counter = 0
                alert_placeholder.empty()

    frame_placeholder.image(frame_rgb, channels="RGB")

cap.release()
