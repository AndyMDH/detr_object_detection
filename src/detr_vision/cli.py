"""
Command-line interface module for the DETR Vision project.
"""
import argparse
import os
from typing import Dict, Any


def create_image_detection_parser() -> argparse.ArgumentParser:
    """
    Create a parser for image detection command-line arguments.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Detect objects in an image using DETR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image"
    )

    # Optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/detr-resnet-50",
        help="DETR model to use"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for detections"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output image. If not specified, will use 'data/outputs/result_<filename>'"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to run the model on (cpu or cuda). If not specified, will use CUDA if available."
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the detection results"
    )

    return parser


def create_webcam_detection_parser() -> argparse.ArgumentParser:
    """
    Create a parser for webcam detection command-line arguments.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Detect objects in webcam feed using DETR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/detr-resnet-50",
        help="DETR model to use"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections"
    )

    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera ID to use"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to run the model on (cpu or cuda). If not specified, will use CUDA if available."
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory to save captured frames with detections"
    )

    return parser


def parse_args(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    """
    Parse command-line arguments and perform basic validation.

    Args:
        parser: Argument parser to use

    Returns:
        Dictionary of parsed arguments
    """
    args = parser.parse_args()

    # Convert arguments to a dictionary
    args_dict = vars(args)

    # Additional validation could be added here

    return args_dict