import streamlit as st
import cv2
import numpy as np
import os
import pickle
import logging
from typing import Tuple, List, Dict
from mtcnn import MTCNN
from keras_facenet import FaceNet
from PIL import Image
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Face Recognition System",
    page_icon=":camera:",
    layout="wide"
)

# Load models with caching
@st.cache_resource
def load_models():
    try:
        logger.info("Loading models...")
        detector = MTCNN()
        embedder = FaceNet()
        
        # Load classifier model
        with open("models/facerecognitionDL.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load label encoder
        with open("models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
            
        return detector, embedder, model, label_encoder
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"Failed to load models: {e}")
        st.stop()

detector, embedder, model, label_encoder = load_models()

def recognize_faces(frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
    """Recognize faces in a frame and return results with annotated image"""
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)
        results = []
        annotated_img = img_rgb.copy()

        for face in faces:
            try:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                face_roi = img_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (160, 160))
                
                # Get embedding and predict
                embedding = embedder.embeddings(np.expand_dims(face_resized, axis=0))
                predictions = model.predict(embedding)
                
                if predictions.ndim == 1:
                    pred_prob = predictions
                else:
                    pred_prob = predictions[0]
                
                pred_class = np.argmax(pred_prob)
                confidence = np.max(pred_prob) * 100
                name = "Unknown" if confidence < 40 else label_encoder.inverse_transform([pred_class])[0]
                
                # Draw annotations
                color = (255, 0, 0) if name == "Unknown" else (0, 255, 0)
                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(annotated_img, f"{name} ({confidence:.2f}%)", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, color, 2)
                
                results.append({
                    "name": name,
                    "confidence": float(confidence),
                    "box": [int(x), int(y), int(w), int(h)]
                })
                
            except Exception as e:
                logger.error(f"Error processing face: {e}")
                continue
                
        return results, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return [], frame

def get_video_capture():
    """Get video source with EC2 fallbacks"""
    try:
        # Try camera first
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            return cap
        
        # Fallback to test video
        if os.path.exists("test_video.mp4"):
            return cv2.VideoCapture("test_video.mp4")
            
        return None
    except Exception as e:
        logger.error(f"Video capture error: {e}")
        return None

def generate_synthetic_frame(frame_count: int):
    """Generate placeholder frame when no camera available"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "EC2 Virtual Camera", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Frame: {frame_count}", (50, 290), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return img

def main():
    st.title("Face Recognition System")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Image Recognition", "Live Recognition"])
    
    with tab1:
        st.header("Image Recognition")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            if st.button("Recognize Faces", key="img_recog"):
                with st.spinner("Processing image..."):
                    results, processed_img = recognize_faces(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(processed_img, caption="Recognized Faces", use_container_width=True)
                    
                    if results:
                        st.subheader("Recognition Results")
                        for result in results:
                            st.write(f"**{result['name']}** (Confidence: {result['confidence']:.2f}%)")
                    else:
                        st.warning("No faces detected")
    
    with tab2:
        st.header("Live Recognition")
        # st.warning("Note: On EC2 instances, a virtual camera will be simulated")
        
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0
        if 'stop_camera' not in st.session_state:
            st.session_state.stop_camera = False
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Camera", key="start_cam"):
                st.session_state.stop_camera = False
        with col2:
            if st.button("Stop Camera", key="stop_cam"):
                st.session_state.stop_camera = True
        
        FRAME_WINDOW = st.empty()
        cap = get_video_capture()
        
        while not st.session_state.stop_camera:
            if cap is None or not cap.isOpened():
                frame = generate_synthetic_frame(st.session_state.frame_count)
            else:
                ret, frame = cap.read()
                if not ret:
                    frame = generate_synthetic_frame(st.session_state.frame_count)
            
            # Process every 3rd frame to reduce load
            if st.session_state.frame_count % 3 == 0:
                results, processed_frame = recognize_faces(frame)
                FRAME_WINDOW.image(processed_frame, channels="BGR", use_container_width=True)
            
            st.session_state.frame_count += 1
            time.sleep(0.05)  # Control frame rate
            
        if cap is not None and cap.isOpened():
            cap.release()

if __name__ == "__main__":
    main()