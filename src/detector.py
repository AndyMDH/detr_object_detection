"""
Core object detection functionality using the DETR model
"""
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import os
import time


class ObjectDetector:
    def __init__(self, model_name="facebook/detr-resnet-50", threshold=0.7):
        """
        Initialize the object detector with a pre-trained model

        Args:
            model_name: Name of the pre-trained model
            threshold: Confidence threshold for detections
        """
        self.model_name = model_name
        self.threshold = threshold
        self.device = None
        self.processor = None
        self.model = None
        self.id2label = None

        # Load the model
        self._load_model()

    def _load_model(self):
        """
        Load the detection model and processor
        """
        print(f"Loading model: {self.model_name}")
        start_time = time.time()

        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        try:
            # Load the processor and model
            self.processor = DetrImageProcessor.from_pretrained(self.model_name, revision="no_timm")
            self.model = DetrForObjectDetection.from_pretrained(self.model_name, revision="no_timm")

            # Move model to device
            self.model.to(self.device)

            # Get the label map
            self.id2label = self.model.config.id2label

            load_time = time.time() - start_time
            print(f"Model loaded successfully in {load_time:.2f} seconds")
            print(f"Model can detect {len(self.id2label)} classes")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def detect(self, image):
        """
        Detect objects in an image

        Args:
            image: PIL Image to process

        Returns:
            Dictionary with detection results
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image")

        # Prepare image for the model
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.threshold
        )[0]

        # Convert tensors to Python types
        detections = []

        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            detection = {
                "label": self.id2label[label_id.item()],
                "score": score.item(),
                "box": box.cpu().numpy().tolist()
            }
            detections.append(detection)

        return detections

    def detect_and_count(self, image):
        """
        Detect objects and count occurrences of each class

        Args:
            image: PIL Image to process

        Returns:
            Tuple of (detections, class_counts)
        """
        # Get detections
        detections = self.detect(image)

        # Count occurrences of each class
        class_counts = {}
        for detection in detections:
            label = detection["label"]
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        return detections, class_counts

    def get_available_classes(self):
        """
        Returns the list of classes the model can detect

        Returns:
            List of class names
        """
        return list(self.id2label.values())