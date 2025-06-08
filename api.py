from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import logging
import tempfile
import os
import sys
import whisper
from src.models.detection import ObjectDetector
from src.models.matching import ProductMatcher
from src.processing.vibe import VibeClassifier
from moviepy.video.io.VideoFileClip import VideoFileClip


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fashion Vibe Analyzer API", description="API for processing videos to detect fashion items and classify vibes.")

def crop_image(frame: np.ndarray, bbox: list) -> np.ndarray:
    """
    Crop the frame using the bounding box.
    
    Args:
        frame (np.ndarray): Input frame in RGB format.
        bbox (list): Bounding box in format [x_min, y_min, x_max, y_max], normalized.
    
    Returns:
        np.ndarray: Cropped image, or None if cropping fails.
    """
    try:
        height, width = frame.shape[:2]
        x_min = int(bbox[0] * width)
        y_min = int(bbox[1] * height)
        x_max = int(bbox[2] * width)
        y_max = int(bbox[3] * height)

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)

        if x_max <= x_min or y_max <= y_min:
            logger.warning(f"Invalid bounding box after adjustment: [{x_min}, {y_min}, {x_max}, {y_max}]")
            return None

        cropped = frame[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            logger.warning("Cropped image is empty")
            return None

        return cropped
    except Exception as e:
        logger.warning(f"Cropping failed for bbox: {bbox}, error: {str(e)}")
        return None

def extract_audio_to_wav(video_path: str) -> str:
    """
    Extract audio from a video file and save it as a WAV file.
    
    Args:
        video_path (str): Path to the video file.
    
    Returns:
        str: Path to the extracted WAV file.
    """
    try:
        with VideoFileClip(video_path) as video:
            audio_path = video_path.replace(os.path.splitext(video_path)[1], ".wav")
            video.audio.write_audiofile(audio_path)
            return audio_path
    except Exception as e:
        logger.error(f"Failed to extract audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to extract audio: {str(e)}")

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio using Whisper.
    
    Args:
        audio_path (str): Path to the audio file (WAV).
    
    Returns:
        str: Transcribed text.
    """
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        transcript = result["text"].strip()
        logger.info(f"Transcription successful: {transcript[:50]}...")
        return transcript
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {str(e)}")
        return ""

def process_video(video_path: str, catalog_path: str, captions: str) -> dict:
    """
    Process a video to detect fashion items and classify vibes, including audio transcription.
    
    Args:
        video_path (str): Path to the video file.
        catalog_path (str): Path to the catalog CSV file.
        captions (str): Captions text.
    
    Returns:
        dict: Structured JSON with video results.
    """
    try:
        detector = ObjectDetector()
        matcher = ProductMatcher(catalog_path, "data/product_data.xlsx")
        vibe_classifier = VibeClassifier()
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize models: {str(e)}")

    # Extract audio and transcribe
    audio_path = None
    transcript = ""
    try:
        audio_path = extract_audio_to_wav(video_path)
        transcript = transcribe_audio(audio_path)
        logger.info("Audio transcription completed.")
    except Exception as e:
        logger.warning(f"Audio transcription failed, proceeding with captions only: {str(e)}")
        transcript = ""

    # Combine captions and transcript for vibe classification
    combined_text = captions
    if transcript:
        combined_text = f"{captions}\nTranscript: {transcript}"
    logger.info(f"Combined text for vibe classification: {combined_text[:100]}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video file")
        raise HTTPException(status_code=400, detail="Failed to open video file")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video has {frame_count} frames at {fps} fps")

    frame_interval = 5  # Process every 5th frame
    result = {"video_id": os.path.basename(video_path), "frames": []}
    processed_frames = 0
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % frame_interval == 0:
            logger.info(f"Processing frame {current_frame}/{frame_count}...")
            detections = detector.detect_objects(frame)
            frame_detections = []
            for detection in detections:
                cropped_image = crop_image(frame, detection["bbox"])
                if cropped_image is not None:
                    match_type, product_id, similarity = matcher.match_product(cropped_image)
                    if match_type != "no_match":
                        product_info = matcher.get_product_info(product_id)
                        frame_detections.append({
                            "label": detection["class"],
                            "confidence": float(detection["confidence"]),
                            "match": {
                                "type": match_type,
                                "product_id": product_id,
                                "similarity": similarity,
                                "product_type": product_info["product_type"],
                                "color": product_info["color"]
                            }
                        })
            if frame_detections:
                result["frames"].append({"frame": current_frame, "detections": frame_detections})
            processed_frames += 1

        current_frame += 1

    cap.release()

    # Classify vibes using combined text
    result["vibes"] = vibe_classifier.classify_vibe(combined_text)
    logger.info("Processing complete!")
    
    # Format the result to match the required JSON structure
    formatted_result = {
        "video_id": result["video_id"],
        "vibes": result["vibes"],
        "products": []
    }
    for frame in result["frames"]:
        for detection in frame["detections"]:
            formatted_result["products"].append({
                "type": detection["match"]["product_type"],
                "color": detection["match"]["color"],
                "match_type": detection["match"]["type"],
                "matched_product_id": detection["match"]["product_id"],
                "confidence": detection["match"]["similarity"]
            })
    
    # Clean up audio file
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)

    return formatted_result

@app.post("/process", response_model=dict)
async def process_video_endpoint(
    video: UploadFile = File(...),
    catalog: UploadFile = File(...),
    captions: UploadFile = File(...)
) -> dict:
    """
    Process a video to detect fashion items and classify vibes, including audio transcription.
    
    Args:
        video (UploadFile): Video file (MP4, AVI, MOV).
        catalog (UploadFile): Catalog CSV file.
        captions (UploadFile): Captions text file.
    
    Returns:
        dict: Structured JSON with video results.
    """
    # Validate file types
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid video file type. Must be MP4, AVI, or MOV.")
    if not catalog.content_type == "text/csv":
        raise HTTPException(status_code=400, detail="Invalid catalog file type. Must be CSV.")
    if not captions.content_type == "text/plain":
        raise HTTPException(status_code=400, detail="Invalid captions file type. Must be a text file.")

    # Save uploaded files temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(await video.read())
            video_path = tmp_video.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_catalog:
            tmp_catalog.write(await catalog.read())
            catalog_path = tmp_catalog.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_captions:
            tmp_captions.write(await captions.read())
            captions_path = tmp_captions.name

        # Read captions
        with open(captions_path, 'r', encoding='utf-8') as f:
            captions_text = f.read().strip()

        # Process the video
        result = process_video(video_path, catalog_path, captions_text)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

    finally:
        # Clean up temporary files
        for path in [video_path, catalog_path, captions_path]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)