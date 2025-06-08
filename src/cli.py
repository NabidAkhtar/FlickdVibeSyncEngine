import argparse
import cv2
import numpy as np
import json
import logging
import os
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
from src.models.detection import ObjectDetector
from src.models.matching import ProductMatcher
from src.processing.vibe import VibeClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        raise Exception(f"Failed to extract audio: {str(e)}")

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
        raise Exception(f"Failed to initialize models: {str(e)}")

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
        raise Exception("Failed to open video file")

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

def main():
    parser = argparse.ArgumentParser(description="Fashion Vibe Analyzer CLI")
    parser.add_argument("--video", required=True, help="Path to the video file (MP4, AVI, MOV)")
    parser.add_argument("--catalog", required=True, help="Path to the catalog CSV file")
    parser.add_argument("--captions", required=True, help="Path to the captions text file")
    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.video):
        logger.error("Video file does not exist")
        print("Error: Video file does not exist")
        exit(1)
    if not os.path.exists(args.catalog):
        logger.error("Catalog file does not exist")
        print("Error: Catalog file does not exist")
        exit(1)
    if not os.path.exists(args.captions):
        logger.error("Captions file does not exist")
        print("Error: Captions file does not exist")
        exit(1)

    # Read captions
    with open(args.captions, 'r', encoding='utf-8') as f:
        captions = f.read().strip()

    # Process the video
    try:
        result = process_video(args.video, args.catalog, captions)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        print(f"Error processing video: {str(e)}")
        exit(1)

    # Save the result to a JSON file
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save the result to a JSON file in the outputs directory
    output_file = os.path.join(output_dir, f"output_{os.path.basename(args.video)}.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()