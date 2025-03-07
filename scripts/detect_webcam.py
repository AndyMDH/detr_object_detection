#!/usr/bin/env python
"""
Script to detect objects in webcam feed using DETR.
"""
import os
import cv2
import sys
import time
from pathlib import Path
from datetime import datetime

# Add the parent directory to the Python path to import our package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detr_vision.model import DetrObjectDetector
from src.detr_vision.camera import Camera
from src.detr_vision.visualization import draw_detections, save_image
from src.detr_vision.cli import create_webcam_detection_parser, parse_args


def main():
    """
    Main function for webcam object detection.
    """
    # Parse command-line arguments
    parser = create_webcam_detection_parser()
    args = parse_args(parser)

    # Create the object detector
    detector = DetrObjectDetector(
        model_name=args["model"],
        device=args["device"]
    )

    # Create save directory if specified
    save_path = args["save_path"]
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving frames to: {save_path}")

    # Initialize FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    print(f"Starting webcam detection (press 'q' to quit, 's' to save current frame)...")

    # Open the camera and start detection
    with Camera(
            camera_id=args["camera_id"],
            width=args["width"],
            height=args["height"]
    ) as camera:
        for frame in camera.stream():
            # Update FPS calculation
            fps_frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time > 1.0:  # Update FPS every second
                fps = fps_frame_count / elapsed_time
                fps_frame_count = 0
                fps_start_time = time.time()

            # Detect objects in the frame
            detections = detector.detect(frame, threshold=args["threshold"])

            # Draw detections on the frame
            result_frame = draw_detections(
                frame,
                detections,
                confidence_threshold=args["threshold"]
            )

            # Add FPS display
            cv2.putText(
                result_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # Display the result
            cv2.imshow("DETR Object Detection", result_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            # Quit if 'q' is pressed
            if key == ord('q'):
                break

            # Save the current frame if 's' is pressed
            if key == ord('s') or save_path:
                # Generate timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if save_path:
                    # If save_path is specified, use it as the directory
                    filename = f"detection_{timestamp}.jpg"
                    output_path = os.path.join(save_path, filename)
                else:
                    # Otherwise, save to the default output directory
                    output_path = f"data/outputs/webcam_{timestamp}.jpg"

                save_image(result_frame, output_path)
                print(f"Saved frame to: {output_path}")

    # Clean up
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())