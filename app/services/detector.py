import logging
import cv2
import numpy as np
from ultralytics import YOLO

# Utilize uvicorn's logger so logs output consistently with the web server
logger = logging.getLogger("uvicorn.error")


class YOLOObstacleDetector:
    def __init__(
        self,
        model_path: str = "models/best.pt",
        device: str = "cpu",
        imgsz: int = 1920,
        conf_threshold: float = 0.25,
    ):
        """Initializes and loads the YOLO model into memory."""
        logger.info(f"Loading YOLO model weights from {model_path}...")
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold

        self.model = YOLO(self.model_path)
        logger.info("YOLO model loaded successfully!")

    def predict(self, image_bytes: bytes) -> dict:
        """Takes raw image bytes from an HTTP request, decodes them,

        runs inference, and returns a structured dictionary of results.
        """
        # Convert raw binary bytes into a NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode the array into an OpenCV image (BGR format)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image. Format might be invalid.")

        # Run inference using YOLO
        results = self.model(
            image,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf_threshold,
        )

        # Grab the first result object from the list
        result = results[0]

        # Parse out bounding boxes and metadata
        predictions = []
        for box in result.boxes:
            # Extract coordinates (xmin, ymin, xmax, ymax) as standard Python floats
            xyxy = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            predictions.append(
                {
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": round(confidence, 4),
                    "bbox": [round(coord, 2) for coord in xyxy],
                }
            )

        return {"count": len(predictions), "predictions": predictions}