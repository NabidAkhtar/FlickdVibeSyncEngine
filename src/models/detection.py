import logging
import numpy as np
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import cv2
import torch

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model_name: str = "valentinafeve/yolos-fashionpedia"):
        """
        Initialize YOLOS model for fashion item detection.
        
        Args:
            model_name (str): Hugging Face model name for yolos-fashionpedia.
        """
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_name,
                size={'longest_edge': 800, 'shortest_edge': 600}
            )
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            logger.info("YOLOS-Fashionpedia model loaded successfully")
            
            self.fashion_classes = ['dress','co-ord','jacket','jeans','jumpsuit','playsuit','shirt','shorts','skirt','skorts','top','trouser','t-shirt']
        except Exception as e:
            logger.error(f"Failed to load YOLOS-Fashionpedia model: {str(e)}")
            raise

    def _fix_bbox(self, bbox: list) -> list:
        """
        Convert bounding box from [center_x, center_y, width, height] to [x_min, y_min, x_max, y_max]
        and ensure x_min < x_max and y_min < y_max.
        
        Args:
            bbox (list): Bounding box in format [center_x, center_y, width, height], normalized.
        
        Returns:
            list: Fixed bounding box in format [x_min, y_min, x_max, y_max].
        """
        center_x, center_y, width, height = bbox
        # Convert to [x_min, y_min, x_max, y_max]
        x_min = center_x - (width / 2)
        y_min = center_y - (height / 2)
        x_max = center_x + (width / 2)
        y_max = center_y + (height / 2)

        # Ensure coordinates are within [0, 1]
        x_min = max(0.0, min(1.0, x_min))
        y_min = max(0.0, min(1.0, y_min))
        x_max = max(0.0, min(1.0, x_max))
        y_max = max(0.0, min(1.0, y_max))

        # Fix swapped coordinates
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        return [x_min, y_min, x_max, y_max]

    def detect_objects(self, frame: np.ndarray) -> list:
        """
        Detect fashion items in a frame using yolos-fashionpedia.
        
        Args:
            frame (np.ndarray): Input image frame (BGR format from OpenCV).
        
        Returns:
            list: List of detections with class, bbox, confidence.
                  bbox is in format [x_min, y_min, x_max, y_max], normalized.
        """
        try:
            # Convert BGR (OpenCV) to RGB
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Preprocess image
            inputs = self.image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run detection
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process outputs
            detections = []
            logits = outputs.logits[0]  # Shape: [num_predictions, num_classes]
            boxes = outputs.pred_boxes[0]  # Shape: [num_predictions, 4]
            scores = torch.softmax(logits, dim=-1)  # Shape: [num_predictions, num_classes]
            
            # Iterate over each prediction
            for i in range(scores.shape[0]):
                max_score, max_idx = scores[i].max(dim=-1)  # Get max score and index for this prediction
                confidence = max_score.item()
                if confidence > 0.5:  # Confidence threshold
                    class_id = max_idx.item()
                    class_name = self.model.config.id2label.get(class_id, "unknown")
                    if class_name in self.fashion_classes:
                        box = boxes[i].cpu().tolist()  # [center_x, center_y, width, height]
                        # Convert and fix bounding box
                        fixed_box = self._fix_bbox(box)
                        detections.append({
                            'class': class_name,
                            'bbox': fixed_box,  # [x_min, y_min, x_max, y_max]
                            'confidence': confidence
                        })
            
            logger.info(f"Detected {len(detections)} fashion items in frame")
            return detections
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            return []