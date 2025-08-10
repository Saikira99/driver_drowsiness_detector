import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import time

# Load Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# App Title
st.set_page_config(page_title="Face Detection", page_icon="📷")
st.title("📷 Real-Time Face Detection (OpenCV + Streamlit)")

# Sidebar controls
st.sidebar.header("Settings")
show_fps = st.sidebar.checkbox("Show FPS", value=True)
play_alert = st.sidebar.checkbox("Play Sound on Detection", value=True)

# JavaScript sound alert
alert_sound = """
<script>
function playAlert() {
    var audio = new Audio("https://www.soundjay.com/buttons/sounds/beep-07.mp3");
    audio.play();
}
</script>
"""

# Face detection transformer
class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_time = time.time()
        self.fps = 0
        self.face_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        self.face_count = len(faces)

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # FPS calculation
        if show_fps:
            now = time.time()
            self.fps = 1 / (now - self.last_time)
            self.last_time = now
            cv2.putText(img, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img

# Start WebRTC streamer
ctx = webrtc_streamer(
    key="face-detection",
    video_transformer_factory=FaceDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False}
)

# Display detection stats
if ctx.video_transformer:
    st.markdown(f"**Detected Faces:** {ctx.video_transformer.face_count}")

    # Play sound if face detected
    if play_alert and ctx.video_transformer.face_count > 0:
        st.markdown(alert_sound, unsafe_allow_html=True)
        st.markdown("<script>playAlert()</script>", unsafe_allow_html=True)
