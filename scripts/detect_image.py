#!/usr/bin/env python
"""
Script to detect objects in an image using DETR.
"""
import os
import cv2
import sys
from pathlib import Path

# Add the parent directory to the Python path to import our package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detr_vision.model import DetrObjectDetector
from src.detr_vision.visualization import draw_detections, save_image, show_image
from src.detr_vision.cli import create_image_detection_parser, parse_args


def main():
    """
    Main function for image object detection.
    """
    # Parse command-line arguments
    parser = create_image_detection_parser()
    args = parse_args(parser)

    # Load the image
    image_path = args["image_path"]
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return 1

    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Couldn't read image at {image_path}")
        return 1

    # Create the object detector
    detector = DetrObjectDetector(
        model_name=args["model"],
        device=args["device"]
    )

    # Run object detection
    print("Detecting objects...")
    detections = detector.detect(image, threshold=args["threshold"])

    # Draw detections on the image
    result_image = draw_detections(image, detections, confidence_threshold=0.0)

    # Print detection results
    print(f"Found {len(detections['boxes'])} objects:")
    for label, score in zip(detections["labels"], detections["scores"]):
        print(f"  - {label}: {score:.2f}")

    # Save the image if requested
    if args["output"] is not None:
        output_path = args["output"]
    else:
        # Create a default output path
        filename = os.path.basename(image_path)
        output_dir = "data/outputs"
        output_path = os.path.join(output_dir, f"result_{filename}")

    saved_path = save_image(result_image, output_path)
    print(f"Saved result to: {saved_path}")

    # Display the image if requested
    if args["display"]:
        print("Displaying result (press any key to continue)...")
        show_image(result_image)

    return 0


if __name__ == "__main__":
    sys.exit(main())