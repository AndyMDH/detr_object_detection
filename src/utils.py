"""
Utility functions for the object detection project
"""
import os
import requests
from PIL import Image
import cv2
import numpy as np


def download_image(url, save_path):
    """
    Download an image from a URL and save it to disk

    Args:
        url: URL of the image
        save_path: Path to save the image

    Returns:
        Path to the saved image
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        # Download the image
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses

        # Save the image
        with open(save_path, "wb") as f:
            f.write(response.content)

        print(f"Image downloaded and saved to {save_path}")
        return save_path

    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def load_image(image_path):
    """
    Load an image from a file path

    Args:
        image_path: Path to the image

    Returns:
        PIL Image object
    """
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def pil_to_cv2(pil_image):
    """
    Convert a PIL Image to an OpenCV image (numpy array)

    Args:
        pil_image: PIL Image object

    Returns:
        OpenCV image (numpy array in BGR format)
    """
    # Convert PIL Image to RGB numpy array
    rgb_array = np.array(pil_image)

    # Convert RGB to BGR (OpenCV format)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    return bgr_array


def cv2_to_pil(cv2_image):
    """
    Convert an OpenCV image to a PIL Image

    Args:
        cv2_image: OpenCV image (numpy array in BGR format)

    Returns:
        PIL Image object
    """
    # Convert BGR to RGB
    rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_array)

    return pil_image


def save_image(image, save_path):
    """
    Save an image to disk

    Args:
        image: PIL Image or OpenCV image
        save_path: Path to save the image

    Returns:
        Path to the saved image
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        # Check if it's a PIL Image
        if isinstance(image, Image.Image):
            image.save(save_path)
        # Check if it's an OpenCV image (numpy array)
        elif isinstance(image, np.ndarray):
            cv2.imwrite(save_path, image)
        else:
            raise TypeError("Image must be a PIL Image or OpenCV image (numpy array)")

        print(f"Image saved to {save_path}")
        return save_path

    except Exception as e:
        print(f"Error saving image: {e}")
        return None


def get_sample_image(filename="sample.jpg"):
    """
    Get a sample image for testing. Downloads a sample image if it doesn't exist.

    Args:
        filename: Name to save the sample image as

    Returns:
        Path to the sample image
    """
    # Path to save the image
    save_dir = os.path.join(os.getcwd(), "images")
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, filename)

    # Check if the image already exists
    if not os.path.exists(image_path):
        # URL for a sample image (a dog from COCO dataset)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"

        # Download the image
        return download_image(url, image_path)
    else:
        print(f"Using existing image at {image_path}")
        return image_path