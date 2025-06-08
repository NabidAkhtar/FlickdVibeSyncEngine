import streamlit as st
import cv2
import numpy as np
import json
import logging
import tempfile
import os
import sys
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
from src.models.detection import ObjectDetector
from src.models.matching import ProductMatcher
from src.processing.vibe import VibeClassifier


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Advanced CSS for an awesome and interesting UI
st.markdown("""
    <style>
        /* Global Styles */
        .main {
            background: linear-gradient(135deg, #1f1c2c 0%, #928dab 100%);
            font-family: 'Montserrat', sans-serif;
            color: #fff;
            padding: 20px;
        }
        /* Title */
        h1 {
            color: #ff6f61;
            font-size: 3.5rem;
            font-weight: 800;
            text-align: center;
            margin-top: 30px;
            text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3);
            animation: fadeInDown 1s ease-in-out;
        }
        /* Subheader */
        h3 {
            color: #feca57;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 20px;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
        }
        /* Tagline */
        .tagline {
            color: #dfe6e9;
            font-size: 1.3rem;
            text-align: center;
            margin-bottom: 30px;
            animation: fadeIn 1.5s ease-in-out;
        }
        /* Container for File Upload Section */
        .upload-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            margin: 20px 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        /* File Uploader */
        .stFileUploader {
            border: 3px dashed #ff6f61;
            border-radius: 15px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
        }
        .stFileUploader:hover {
            border-color: #feca57;
            background: rgba(255, 255, 255, 0.1);
            transform: scale(1.02);
        }
        .stFileUploader label {
            color: #dfe6e9 !important;
            font-weight: 600 !important;
        }
        /* Analyze Button */
        .stButton>button {
            background: linear-gradient(90deg, #ff6f61, #feca57);
            color: #fff;
            border: none;
            border-radius: 30px;
            padding: 15px 40px;
            font-size: 1.2rem;
            font-weight: 700;
            transition: all 0.4s ease;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            animation: pulse 2s infinite;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #feca57, #ff6f61);
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        }
        /* Result Card */
        .result-card {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
            transition: transform 0.4s ease;
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(8px);
            animation: fadeInUp 0.8s ease-in-out;
        }
        .result-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
        }
        /* Frame Title in Result Card */
        .frame-title {
            color: #feca57;
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 15px;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
        }
        /* Detection Details */
        .detection-details {
            color: #dfe6e9;
            font-size: 1.1rem;
            line-height: 1.8;
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        /* Vibe Section */
        .vibe-section {
            background: linear-gradient(90deg, #ff6f61, #feca57);
            color: #fff;
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            font-size: 1.4rem;
            font-weight: 600;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
            animation: bounceIn 1s ease-in-out;
        }
        /* Spinner */
        .stSpinner {
            color: #feca57;
        }
        /* Success Message */
        .stSuccess {
            background: rgba(46, 204, 113, 0.2);
            color: #2ecc71;
            border-radius: 15px;
            padding: 20px;
            font-weight: 600;
            border: 1px solid #2ecc71;
        }
        /* Error Message */
        .stError {
            background: rgba(231, 76, 60, 0.2);
            color: #e74c3c;
            border-radius: 15px;
            padding: 20px;
            font-weight: 600;
            border: 1px solid #e74c3c;
        }
        /* Progress Bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #ff6f61, #feca57);
            border-radius: 15px;
        }
        /* Video Player */
        video {
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            margin: 30px 0;
            transition: transform 0.3s ease;
        }
        video:hover {
            transform: scale(1.02);
        }
        /* Animations */
        @keyframes fadeInDown {
            0% { opacity: 0; transform: translateY(-20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        @keyframes fadeInUp {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes bounceIn {
            0% { transform: scale(0.3); opacity: 0; }
            50% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(1); }
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 111, 97, 0.7); }
            70% { box-shadow: 0 0 0 20px rgba(255, 111, 97, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 111, 97, 0); }
        }
        /* Column Styling */
        .stColumn {
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Add Google Fonts for a bold, modern typography
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

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

def process_video(video_path, catalog_path, captions):
    try:
        detector = ObjectDetector()
        matcher = ProductMatcher(catalog_path, "data/product_data.xlsx")
        vibe_classifier = VibeClassifier()
    except Exception as e:
        st.error(f"Failed to initialize models: {str(e)}")
        return None

    # Extract audio and transcribe
    audio_path = None
    transcript = ""
    try:
        audio_path = extract_audio_to_wav(video_path)
        transcript = transcribe_audio(audio_path)
        logger.info("Audio transcription completed.")
    except Exception as e:
        logger.warning(f"Audio transcription failed, proceeding with captions only: {str(e)}")
        st.warning("Audio transcription failed. Proceeding with captions only.")
        transcript = ""

    # Combine captions and transcript for vibe classification
    combined_text = captions
    if transcript:
        combined_text = f"{captions}\nTranscript: {transcript}"
    logger.info(f"Combined text for vibe classification: {combined_text[:100]}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video file")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video has {frame_count} frames at {fps} fps")

    frame_interval = 5  # Process every 5th frame
    result = {"video": video_path, "frames": []}
    processed_frames = 0
    current_frame = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % frame_interval == 0:
            logger.info(f"Processing frame {current_frame}/{frame_count}...")
            status_text.text(f"Processing frame {current_frame}/{frame_count}...")
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
            progress_bar.progress(min(current_frame / frame_count, 1.0))

        current_frame += 1

    cap.release()

    result["vibes"] = vibe_classifier.classify_vibe(combined_text)
    status_text.text("Processing complete!")

    # Save the result as a JSON file
    with open(f"evaluation_{os.path.basename(video_path)}.json", "w") as f:
        json.dump(result, f, indent=4)

    # Clean up audio file
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)

    return result

def main():
    # Header with a Dynamic Title
    st.markdown("""
        <div style="text-align: center;">
            <h1>🔥 Fashion Vibe Analyzer</h1>
            <p class="tagline">Unleash the Magic of Fashion Trends with AI-Powered Insights!</p>
        </div>
    """, unsafe_allow_html=True)

    # File Upload Section
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("Drop Your Files Here!")
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
        with col1:
            video_file = st.file_uploader("Video File", type=["mp4", "avi", "mov"], help="Upload a video file (MP4, AVI, MOV)")
        with col2:
            catalog_file = st.file_uploader("Catalog CSV", type=["csv"], help="Upload the catalog CSV file")
        with col3:
            captions_file = st.file_uploader("Captions File", type=["txt"], help="Upload the captions text file")
        st.markdown('</div>', unsafe_allow_html=True)

    # Analyze Button with Centered Alignment
    st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
    analyze_button = st.button("Analyze Now 🚀")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_button:
        if video_file and catalog_file and captions_file:
            # Save files temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(video_file.read())
                video_path = tmp_video.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_catalog:
                tmp_catalog.write(catalog_file.read())
                catalog_path = tmp_catalog.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_captions:
                tmp_captions.write(captions_file.read())
                captions_path = tmp_captions.name

            # Read captions
            with open(captions_path, 'r', encoding='utf-8') as f:
                captions = f.read().strip()

            # Process video
            with st.spinner("Analyzing your video... Let the magic happen! ✨"):
                result = process_video(video_path, catalog_path, captions)

            if result:
                # Display Results
                st.success("Analysis Complete! Let's Dive into the Results! 🎉")
                st.video(video_path)

                st.subheader("🎨 Detected Fashion Items")
                for frame in result["frames"]:
                    with st.container():
                        st.markdown(f'<div class="result-card"><div class="frame-title">Frame {frame["frame"]}</div>', unsafe_allow_html=True)
                        for detection in frame["detections"]:
                            with st.container():
                                st.markdown(f"""
                                    <div class="detection-details">
                                        <b>{detection['label']}</b> (Confidence: {detection['confidence']:.2f})<br>
                                        Match: {detection['match']['type']} (ID: {detection['match']['product_id']})<br>
                                        Similarity: {detection['match']['similarity']:.2f}<br>
                                        Product Type: {detection['match']['product_type']}, Color: {detection['match']['color']}
                                    </div>
                                """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                st.subheader("🌟 Vibe Classification")
                st.markdown(f'<div class="vibe-section">{", ".join(result["vibes"])}</div>', unsafe_allow_html=True)

                # Clean up temporary files
                os.remove(video_path)
                os.remove(catalog_path)
                os.remove(captions_path)
            else:
                st.error("Oops! Something went wrong. Please check your files and try again. 🛠️")
        else:
            st.error("Please upload all required files to start the analysis! 📂")

    # Footer with a Fun Vibe
    st.markdown("""
        <div style="text-align: center; margin-top: 50px; color: #dfe6e9;">
            <p>Built with ❤️ for FLICKD AI HACKATHON <p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
