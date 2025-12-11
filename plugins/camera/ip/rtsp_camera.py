"""
rtsp_camera.py

This module provides an RTSPCamera implementation based on the abstract
Camera base class. It supports connecting to an RTSP stream, fetching
RGB frames, applying camera configuration, and safely disconnecting.
"""

import cv2
import datetime
from typing import Optional, Dict, Any

from devices.cameras.base import CameraBase
from devices.cameras.dataclass import RGBFrameData


class RTSPCamera(CameraBase):
    """
    RTSP camera implementation.

    Connects to an RTSP video stream, fetches RGB frames, and allows
    dynamic configuration updates (resolution, brightness, contrast).
    """

    def __init__(self, cam_id: str, config: Dict[str, Any]) -> None:
        """
        Initialize the RTSPCamera instance.

        Args:
            cam_id (str): Unique identifier for the camera.
            config (Dict[str, Any]): Camera configuration, must include
                'stream_url' for the RTSP source.
        """
        super().__init__(cam_id, config)
        self.cap: Optional[cv2.VideoCapture] = None

    def connect(self) -> bool:
        """
        Establish connection to the RTSP stream.

        Returns:
            bool: True if connection succeeded, False otherwise.
        """
        url = self.config.get("stream_url",None)
        if url is None:
            return False

        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            return False

        self._apply_config(self.config)
        self.connected = True
        return True

    def disconnect(self) -> None:
        """
        Safely disconnect from the RTSP stream and release resources.
        """
        self.connected = False
        if self.cap is not None:
            self.cap.release()
        self.cap = None

    def fetch_frame(self) -> Optional[RGBFrameData]:
        """
        Capture a single RGB frame from the RTSP stream.

        Returns:
            Optional[RGBFrameData]: Captured RGB frame, or None if
            connection is lost or frame could not be read.
        """
        if not self.cap or not self.connected:
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.connected = False
            return None

        return RGBFrameData(
            device_id=self.camera_id,
            frame=frame,
            timestamp=datetime.datetime.now(),
            resolution=(frame.shape[1], frame.shape[0]),
            fps=int(self.cap.get(cv2.CAP_PROP_FPS)),
            frame_number=int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
            device_name=self.config.get("name", f"RTSP-{self.camera_id}"),
            is_color=True
        )

    def _apply_config(self, updates: Dict[str, Any]) -> None:
        """
        Apply configuration updates to the RTSP camera.

        Args:
            updates (Dict[str, Any]): Dictionary of configuration
                updates. Supported keys: 'resolution', 'brightness', 'contrast'.

        Raises:
            ValueError: If a property cannot be set on the camera.
        """
        if not self.cap or not self.cap.isOpened():
            return RuntimeError(f"Camera {self.camera_id} is not opened")

        resolution = updates.get("resolution")
        if resolution and isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            width, height = resolution
            if not self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width):
                return ValueError(f"Failed to set width={width} for camera {self.camera_id}")
            if not self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height):
                return ValueError(f"Failed to set height={height} for camera {self.camera_id}")

        brightness = updates.get("brightness")
        if brightness is not None:
            if not self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness):
                return ValueError(f"Failed to set brightness={brightness} for camera {self.camera_id}")

        contrast = updates.get("contrast")
        if contrast is not None:
            if not self.cap.set(cv2.CAP_PROP_CONTRAST, contrast):
                return ValueError(f"Failed to set contrast={contrast} for camera {self.camera_id}")
