"""
Visualization module for displaying and saving detection results.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import random

# Create a persistent color mapping for class labels
COLOR_MAP = {}
BOX_THICKNESS = 2
TEXT_THICKNESS = 2
FONT_SCALE = 0.5


def get_color(label: str) -> Tuple[int, int, int]:
    """
    Get a consistent color for a given label.

    Args:
        label: The class label

    Returns:
        BGR color tuple
    """
    if label not in COLOR_MAP:
        # Generate random color and ensure it's visually distinct
        color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        COLOR_MAP[label] = color

    return COLOR_MAP[label]


def draw_detections(
        image: np.ndarray,
        detections: Dict,
        confidence_threshold: float = 0.0
) -> np.ndarray:
    """
    Draw detection results on an image.

    Args:
        image: Input image as numpy array (OpenCV format)
        detections: Detection results from the model
        confidence_threshold: Minimum confidence to display a detection

    Returns:
        Image with detections drawn on it
    """
    # Make a copy of the image to avoid modifying the original
    img_with_detections = image.copy()

    # Get detection data
    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]

    # Draw each detection that meets the confidence threshold
    for box, score, label in zip(boxes, scores, labels):
        if score < confidence_threshold:
            continue

        # Convert box coordinates to integers
        x1, y1, x2, y2 = box.astype(int)

        # Get color for this class
        color = get_color(label)

        # Draw bounding box
        cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Create label text with class name and confidence score
        text = f"{label}: {score:.2f}"

        # Get text size to position it properly
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS
        )

        # Create a filled rectangle for the text background
        cv2.rectangle(
            img_with_detections,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1  # Filled rectangle
        )

        # Add text with white color
        cv2.putText(
            img_with_detections,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            TEXT_THICKNESS
        )

    return img_with_detections


def save_image(image: np.ndarray, output_path: str) -> str:
    """
    Save an image to disk.

    Args:
        image: Image to save
        output_path: Path to save the image to

    Returns:
        Full path to the saved image
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the image
    cv2.imwrite(output_path, image)

    return output_path


def show_image(image: np.ndarray, title: str = "Detection Result", wait: bool = True):
    """
    Display an image using OpenCV.

    Args:
        image: Image to display
        title: Window title
        wait: Whether to wait for a key press before continuing
    """
    cv2.imshow(title, image)

    if wait:
        # Wait for a key press
        cv2.waitKey(0)
        # Close all windows
        cv2.destroyAllWindows()