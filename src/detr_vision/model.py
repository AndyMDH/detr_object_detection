"""
Model handling module for DETR object detection.
"""
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from typing import Dict, List, Tuple, Union
import numpy as np
from PIL import Image


class DetrObjectDetector:
    """
    A class to handle DETR object detection model operations.
    """

    def __init__(self, model_name: str = "facebook/detr-resnet-50", device: str = None):
        """
        Initialize the DETR object detector.

        Args:
            model_name: Name or path of the DETR model to use
            device: Device to run the model on ('cpu' or 'cuda')
        """
        # If no device is specified, use CUDA if available, otherwise use CPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading DETR model '{model_name}' on {self.device}...")

        # Load the model and processor
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)

        # Move model to the specified device (GPU or CPU)
        self.model.to(self.device)

        # Set model to evaluation mode (not training)
        self.model.eval()

        # Get the class names (labels) from the model config
        self.labels = self.model.config.id2label

        print(f"Model loaded successfully with {len(self.labels)} classes!")

    def detect(self,
               image: Union[np.ndarray, Image.Image],
               threshold: float = 0.7) -> Dict:
        """
        Perform object detection on an image.

        Args:
            image: Input image (can be numpy array from OpenCV or PIL Image)
            threshold: Confidence threshold for detections

        Returns:
            Dictionary containing detection results
        """
        # Convert OpenCV image (numpy array) to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image[:, :, ::-1])  # Convert BGR to RGB
        else:
            image_pil = image

        # Prepare image for the model
        inputs = self.processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():  # No need to track gradients for inference
            outputs = self.model(**inputs)

        # Convert outputs to a more usable format
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=[(image_pil.height, image_pil.width)]
        )[0]

        # Return the processed results
        return {
            "boxes": results["boxes"].cpu().numpy(),
            "scores": results["scores"].cpu().numpy(),
            "labels": [self.labels[l.item()] for l in results["labels"]]
        }