import logging
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .cv.pipeline import run_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("catan-cv-service")

app = FastAPI(
    title="Catan Board CV Service",
    description="Microservice to detect Catan boards, hex centers, terrain types, number tokens, and harbor positions from photos/screenshots.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow requests from all origins (including localhost development and file://)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/detect")
async def detect_board(
    image: UploadFile = File(...),
    mode: str = Form("four")
):
    logger.info(f"Received detection request. Mode: {mode}, Filename: {image.filename}")
    
    if mode not in ("four", "six"):
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'four' or 'six'.")
        
    try:
        # Read uploaded image bytes
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=422, detail="Failed to decode uploaded image. Ensure it is a valid image file.")
            
        # Convert BGR (OpenCV default) to RGB (our CV library default)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Run computer vision pipeline
        result = run_pipeline(img_rgb, mode)
        return result
        
    except HTTPException as he:
        raise he
    except FileNotFoundError as fe:
        logger.error(f"Template loading error: {str(fe)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server template error: {str(fe)}")
    except Exception as e:
        logger.error(f"Error during board detection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Board detection failed: {str(e)}")
