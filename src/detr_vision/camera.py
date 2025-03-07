"""
Camera handling module for webcam access and image processing.
"""
import cv2
import time
from typing import Tuple, Generator, Optional


class Camera:
    """
    A class to handle webcam operations.
    """

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Initialize the camera.

        Args:
            camera_id: ID of the camera to use (usually 0 for built-in webcam)
            width: Desired width of the camera feed
            height: Desired height of the camera feed
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None

    def __enter__(self):
        """
        Open the camera when entering a context manager.
        This lets you use 'with Camera() as cam:' in your code.
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close the camera when exiting a context manager.
        """
        self.close()

    def open(self):
        """
        Open the camera connection.

        Returns:
            self for method chaining
        """
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with ID {self.camera_id}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Give the camera time to warm up
        time.sleep(0.5)

        return self

    def close(self):
        """
        Close the camera connection.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        """
        Read a single frame from the camera.

        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            raise RuntimeError("Camera is not opened. Call open() first.")

        return self.cap.read()

    def stream(self) -> Generator[cv2.typing.MatLike, None, None]:
        """
        Stream frames from the camera.

        Yields:
            Camera frames as they become available
        """
        if self.cap is None:
            self.open()

        while True:
            success, frame = self.cap.read()
            if not success:
                break

            yield frame