import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

def extract_frames(video_path: str, interval: float = 5.0) -> list:
    """
    Extract keyframes from a video at specified intervals.
    
    Args:
        video_path (str): Path to video file.
        interval (float): Interval in seconds between frames.
    
    Returns:
        list: List of frame arrays.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        frames = []
        count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frames.append(frame)
            count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return []