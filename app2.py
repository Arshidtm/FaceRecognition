import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Title
st.title("Camera Access Demo")

# Check if running locally or on EC2
def is_running_on_ec2():
    """Check if the code is running on an EC2 instance"""
    try:
        with open('/sys/hypervisor/uuid', 'r') as f:
            uuid = f.read()
            return uuid.startswith('ec2')
    except:
        return False

ON_EC2 = is_running_on_ec2()

st.write(f"Running on EC2: {ON_EC2}")

if ON_EC2:
    st.warning("""
    You're running on an EC2 instance which typically doesn't have direct camera access.
    The app will use WebRTC to access your local camera through the browser.
    """)

# Option 1: Direct camera access (works locally)
if not ON_EC2:
    st.header("Direct Camera Access (Local Only)")
    run_direct = st.checkbox("Use direct camera access (local only)", value=True)
    
    if run_direct:
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            st.error("Cannot open camera. Please check if it's connected and available.")
        else:
            stop_button = st.button("Stop Direct Camera")
            
            while not stop_button:
                ret, frame = camera.read()
                
                if not ret:
                    st.error("Failed to grab frame. Camera may be disconnected.")
                    break
                
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame)
                except cv2.error as e:
                    st.error(f"Error processing frame: {str(e)}")
                    break
                    
                stop_button = st.button("Stop Direct Camera")

            camera.release()
            st.write("Camera stopped")

# Option 2: WebRTC for remote access
st.header("WebRTC Camera Access (Works Remotely)")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Optional: Add any image processing here
        # For example, convert to grayscale:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if not webrtc_ctx.state.playing:
    st.info("Click 'Start' to enable camera access. You may need to grant permissions.")