"""
Tests for the model module.
"""
import unittest
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path to import our package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detr_vision.model import DetrObjectDetector


class TestDetrObjectDetector(unittest.TestCase):
    """
    Test cases for the DetrObjectDetector class.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a tiny test image (3x3 pixels)
        self.test_image = Image.new('RGB', (3, 3), color='red')

        # Convert to numpy array (OpenCV format)
        self.test_image_cv = np.array(self.test_image)

        # We'll use CPU for tests to ensure they run everywhere
        self.detector = DetrObjectDetector(device="cpu")

    def test_initialization(self):
        """
        Test that the detector initializes correctly.
        """
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.processor)
        self.assertEqual(self.detector.device, "cpu")

    def test_detect_pil_image(self):
        """
        Test detection on a PIL image.
        """
        try:
            results = self.detector.detect(self.test_image, threshold=0.0)

            # Check that the results have the expected structure
            self.assertIn("boxes", results)
            self.assertIn("scores", results)
            self.assertIn("labels", results)

            # Since this is a tiny test image, we don't expect any real detections
            # But the function should still run without errors
        except Exception as e:
            self.fail(f"Detection on PIL image raised exception: {e}")

    def test_detect_cv_image(self):
        """
        Test detection on an OpenCV image (numpy array).
        """
        try:
            results = self.detector.detect(self.test_image_cv, threshold=0.0)

            # Check that the results have the expected structure
            self.assertIn("boxes", results)
            self.assertIn("scores", results)
            self.assertIn("labels", results)

            # Since this is a tiny test image, we don't expect any real detections
            # But the function should still run without errors
        except Exception as e:
            self.fail(f"Detection on OpenCV image raised exception: {e}")


if __name__ == "__main__":
    unittest.main()