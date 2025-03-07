"""
Visualization utilities for object detection
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2


def get_color_mapping(class_list):
    """
    Create a color mapping for classes

    Args:
        class_list: List of class names

    Returns:
        Dictionary mapping class names to RGB color tuples
    """
    # Use a colormap to generate distinct colors
    cmap = plt.cm.get_cmap('hsv', len(class_list))

    colors = {}
    for i, class_name in enumerate(class_list):
        # Get color from colormap and convert to RGB tuple (0-255)
        rgba = cmap(i)
        rgb = tuple(int(255 * c) for c in rgba[:3])
        colors[class_name] = rgb

    return colors


def draw_detections(image, detections, color_mapping=None):
    """
    Draw bounding boxes and labels on an image

    Args:
        image: PIL Image to draw on
        detections: List of detection dictionaries
        color_mapping: Dictionary mapping class names to RGB color tuples

    Returns:
        PIL Image with detections drawn
    """
    # Create a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    # Create a color mapping if not provided
    if color_mapping is None:
        class_list = list(set(d["label"] for d in detections))
        color_mapping = get_color_mapping(class_list)

    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    # Draw each detection
    for detection in detections:
        label = detection["label"]
        score = detection["score"]
        box = detection["box"]

        # Get color for this class
        color = color_mapping.get(label, (255, 0, 0))  # Default to red if class not found

        # Convert box coordinates to integers
        x1, y1, x2, y2 = [int(coord) for coord in box]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label and score
        text = f"{label}: {score:.2f}"

        # Calculate text size
        if hasattr(font, "getbbox"):
            # For newer PIL versions
            text_bbox = font.getbbox(text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        else:
            # Fall back for older PIL versions
            text_width, text_height = font.getsize(text)

        # Draw text background
        draw.rectangle([x1, y1, x1 + text_width, y1 + text_height], fill=color)

        # Draw text
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)

    return draw_image


def draw_detections_cv2(cv2_image, detections, color_mapping=None):
    """
    Draw bounding boxes and labels on an OpenCV image

    Args:
        cv2_image: OpenCV image (numpy array in BGR format)
        detections: List of detection dictionaries
        color_mapping: Dictionary mapping class names to RGB color tuples

    Returns:
        OpenCV image with detections drawn
    """
    # Create a copy of the image to draw on
    draw_image = cv2_image.copy()

    # Create a color mapping if not provided
    if color_mapping is None:
        class_list = list(set(d["label"] for d in detections))
        color_mapping = get_color_mapping(class_list)

    # Draw each detection
    for detection in detections:
        label = detection["label"]
        score = detection["score"]
        box = detection["box"]

        # Get color for this class, convert from RGB to BGR for OpenCV
        rgb_color = color_mapping.get(label, (255, 0, 0))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

        # Convert box coordinates to integers
        x1, y1, x2, y2 = [int(coord) for coord in box]

        # Draw box
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), bgr_color, 2)

        # Draw label and score
        text = f"{label}: {score:.2f}"

        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw text background
        cv2.rectangle(draw_image, (x1, y1), (x1 + text_width, y1 - text_height - 5), bgr_color, -1)

        # Draw text
        cv2.putText(draw_image, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)

    return draw_image


def plot_detections(image, detections, figsize=(10, 10)):
    """
    Plot an image with detections using matplotlib

    Args:
        image: PIL Image
        detections: List of detection dictionaries
        figsize: Figure size

    Returns:
        None (displays the plot)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Display the image
    ax.imshow(np.array(image))

    # Create a color mapping
    class_list = list(set(d["label"] for d in detections))
    color_mapping = get_color_mapping(class_list)

    # Draw each detection
    for detection in detections:
        label = detection["label"]
        score = detection["score"]
        box = detection["box"]

        # Get color for this class, convert to matplotlib format (0-1)
        rgb_color = color_mapping.get(label, (255, 0, 0))
        mpl_color = tuple(c / 255 for c in rgb_color)

        # Extract coordinates
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # Draw box
        rect = plt.Rectangle((x1, y1), width, height,
                             fill=False, edgecolor=mpl_color, linewidth=2)
        ax.add_patch(rect)

        # Draw label and score
        text = f"{label}: {score:.2f}"
        ax.text(x1, y1, text, fontsize=12,
                bbox=dict(facecolor=mpl_color, alpha=0.8, pad=0),
                color='white')

    # Hide axes
    ax.axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()