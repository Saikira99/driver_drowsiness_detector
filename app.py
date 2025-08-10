import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import numpy as np
import playsound
import threading

# EAR calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
alarm_on = False
COUNTER = 0

def sound_alarm():
    playsound.playsound("alarm.mp3")

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        global COUNTER, alarm_on
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = img.shape
                landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

                left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
                right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]

                leftEAR = eye_aspect_ratio(left_eye)
                rightEAR = eye_aspect_ratio(right_eye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < EAR_THRESHOLD:
                    COUNTER += 1
                    if COUNTER >= EAR_CONSEC_FRAMES:
                        if not alarm_on:
                            alarm_on = True
                            threading.Thread(target=sound_alarm).start()
                        cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    COUNTER = 0
                    alarm_on = False

                # Draw eyes
                for (x, y) in left_eye + right_eye:
                    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        return img

st.title("🚗 Driver Drowsiness Detection")
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
