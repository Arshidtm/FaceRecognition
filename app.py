import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os
import uuid
import pickle
import logging
from typing import Tuple, List, Dict
import shutil
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Face Recognition App",
    page_icon=":camera:",
    layout="wide"
)

# Initialize models (cached to load only once)
@st.cache_resource
def load_models():
    try:
        logger.info("Initializing face detection and embedding models...")
        detector = MTCNN()
        embedder = FaceNet()
        
        # Load trained classifier model
        MODEL_PATH = os.path.join("models", "facerecognitionDL.pkl")
        logger.info(f"Loading classifier model from {MODEL_PATH}...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        
        # Load label encoder
        ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
        logger.info(f"Loading label encoder from {ENCODER_PATH}...")
        with open(ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        
        logger.info("All models loaded successfully!")
        return detector, embedder, model, label_encoder

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"Failed to load models: {e}")
        st.stop()

detector, embedder, model, label_encoder = load_models()

# Temporary storage for uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def recognize_face(img: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
    """
    Recognize faces in an image and return recognition results with annotated image.
    """
    try:
        logger.info("Starting face recognition...")
        
        # Convert image to RGB (from BGR if coming from OpenCV)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        logger.debug("Detecting faces...")
        faces = detector.detect_faces(img_rgb)
        if not faces:
            logger.warning("No faces detected in the image")
            return [], img_rgb
            
        results = []
        annotated_img = img_rgb.copy()
        
        for i, face in enumerate(faces):
            try:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                
                # Extract face ROI
                logger.debug(f"Processing face {i+1}/{len(faces)}")
                face_roi = img_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (160, 160))
                
                # Get embedding and predict
                embedding = embedder.embeddings(np.expand_dims(face_resized, axis=0))
                predictions = model.predict(embedding)
                
                # Handle different model output types
                if predictions.ndim == 1:  # For models that output class probabilities directly
                    pred_prob = predictions
                else:  # For models that output one-hot encoded predictions
                    pred_prob = predictions[0]
                
                pred_class = np.argmax(pred_prob)
                confidence = np.max(pred_prob) * 100
                name = "Unknown" if confidence < 40 else label_encoder.inverse_transform([pred_class])[0]
                
                # Draw rectangle and label
                color = (255, 0, 0) if name == "Unknown" else (0, 255, 0)
                logger.info(f"Detected: {name} (confidence: {confidence:.2f}%)")
                
                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(annotated_img, f"{name} ({confidence:.2f}%)", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.9, color, 2)
                
                results.append({
                    "name": name,
                    "confidence": float(confidence),
                    "box": [int(x), int(y), int(w), int(h)]
                })
                
            except Exception as face_error:
                logger.error(f"Error processing face {i+1}: {str(face_error)}")
                continue
                
        return results, annotated_img
        
    except Exception as e:
        logger.error(f"Face recognition failed: {str(e)}", exc_info=True)
        raise

def clear_upload_folder():
    """Empty the upload folder"""
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): 
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
        logger.info("Upload folder cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing upload folder: {e}")

# Main App
def main():
    st.title("Face Recognition System")
    st.write("Upload an image or use live camera for face recognition")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Image Upload", "Live Camera"])
    
    with tab1:
        st.header("Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Read image file
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Display original image
                st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)
                
                # Process image when button is clicked
                if st.button("Recognize Faces", key="recognize_upload"):
                    with st.spinner("Processing image..."):
                        results, annotated_img = recognize_face(img)
                        
                        # Display results
                        st.image(annotated_img, channels="RGB", caption="Recognized Faces", use_container_width=True)
                        
                        if results:
                            st.subheader("Recognition Results")
                            for result in results:
                                st.write(f"- **{result['name']}** (confidence: {result['confidence']:.2f}%)")
                        else:
                            st.warning("No faces detected in the image")
                            
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    with tab2:
        st.header("Live Camera Recognition")
        st.warning("Note: Live recognition requires camera access and may not work in all environments.")
        
        # Initialize webcam state
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        
        # Start/Stop camera button
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.camera_active:
                if st.button("Start Camera", key="start_camera"):
                    st.session_state.camera_active = True
                    st.rerun()
        
        # Camera feed placeholder
        camera_placeholder = st.empty()
        
        if st.session_state.camera_active:
            with col2:
                if st.button("Stop Camera", key="stop_camera"):
                    st.session_state.camera_active = False
                    st.rerun()
            
            # Initialize camera
            cap = cv2.VideoCapture(0)
            
            try:
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame from camera")
                        break
                    
                    # Process every 5th frame to reduce computation
                    if 'frame_count' not in st.session_state:
                        st.session_state.frame_count = 0
                    
                    if st.session_state.frame_count % 5 == 0:
                        results, processed_frame = recognize_face(frame)
                        frame = processed_frame
                    
                    # Display the frame
                    camera_placeholder.image(frame, channels="BGR", use_container_width=True)
                    
                    # Update frame count
                    st.session_state.frame_count += 1
                    
            finally:
                cap.release()
                if not st.session_state.camera_active:
                    camera_placeholder.empty()
    
    # Clear uploads when session ends
    if not st.session_state.get('initialized', False):
        clear_upload_folder()
        st.session_state.initialized = True

if __name__ == "__main__":
    main()