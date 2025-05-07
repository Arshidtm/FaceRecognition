from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static",StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

try:
    # Initialize face detection and embedding models
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

except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise
except pickle.PickleError as e:
    logger.error(f"Error unpickling model: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error during model loading: {e}")
    raise
    
# Temporary storage for uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def recognize_face(image_path: str) -> Tuple[List[Dict], str]:
    """
    Recognize faces in an image and return recognition results with annotated image.
    
    Args:
        image_path (str): Path to the input image file
        
    Returns:
        Tuple containing:
        - List of recognition results (each with name, confidence, bounding box)
        - Path to the output annotated image
        
    Raises:
        FileNotFoundError: If input image doesn't exist
        ValueError: If no faces detected or image processing fails
        Exception: For other unexpected errors
    """
    try:
        logger.info(f"Starting face recognition for image: {image_path}")
        
        # Validate input file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Read and convert image
        logger.debug("Loading and converting image...")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read image file")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        logger.debug("Detecting faces...")
        faces = detector.detect_faces(
            img
        )
        if not faces:
            logger.warning("No faces detected in the image")
            return [], ""
            
        results = []
        for i, face in enumerate(faces):
            try:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                
                # Extract face ROI
                logger.debug(f"Processing face {i+1}/{len(faces)}")
                face_roi = img[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (160, 160))
                
                # Get embedding and predict
                embedding = embedder.embeddings(np.expand_dims(face_resized, axis=0))
                pred_prob = model.predict(embedding)[0]
                pred_class = np.argmax(pred_prob)
                confidence = np.max(pred_prob) * 100
                name = "Unknown" if confidence < 40 else label_encoder.inverse_transform([pred_class])[0]
        
        # Draw rectangle and label (for display)
                color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                
                logger.info(f"Detected: {name} (confidence: {confidence:.2f}%)")
                
                # Draw annotations
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, f"{name} ({confidence:.2f}%)", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.9, (0, 255, 0), 2)
                
                results.append({
                    "name": name,
                    "confidence": float(confidence),
                    "box": [int(x), int(y), int(w), int(h)]
                })
                
            except Exception as face_error:
                logger.error(f"Error processing face {i+1}: {str(face_error)}")
                continue
                
        # Save annotated image
        output_path = os.path.join(UPLOAD_FOLDER, f"result_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved annotated image to: {output_path}")
        
        return results, output_path
        
    except Exception as e:
        logger.error(f"Face recognition failed: {str(e)}", exc_info=True)
        raise
    
def clear_upload_folder():
    """Empty the upload folder while keeping the directory structure"""
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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/recognize")
async def api_recognize(file: UploadFile = File(...)):
    """API endpoint for face recognition"""
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, f"upload_{uuid.uuid4().hex}.jpg")
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process image
        results, output_path = recognize_face(file_path)
        
        # Clean up
        try:
            os.remove(file_path)
            logger.debug(f"Deleted original upload: {file_path}")
        except Exception as delete_error:
            logger.warning(f"Could not delete original file: {delete_error}")
        
        return JSONResponse({
            "results": results,
            "image_url": f"/{output_path}"
        })
    finally:
        # Clear upload folder after every request
        clear_upload_folder()
    
@app.get("/live", response_class=HTMLResponse)
async def live_recognition(request: Request):
    """Page for live face recognition"""
    return templates.TemplateResponse("live.html", {"request": request})