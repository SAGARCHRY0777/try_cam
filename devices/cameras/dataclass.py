"""dataclass.py

Frame data structures for representing images and sensor outputs from various
camera types. Provides base metadata, specialized frame formats, and helpers
for generating diagnostic or error frames. All frame types inherit from
`BaseFrameData` for consistency across the system.
"""

import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple
from abc import ABC

import numpy as np
import cv2

@dataclass
class BaseFrameData(ABC):
    """Base structure for all frame types.

    Attributes:
        device_id (str):
            Identifier of the device or camera that produced the frame.
        frame (np.ndarray):
            Raw image/frame data as a NumPy array.
        timestamp (datetime.datetime):
            Time at which the frame was captured. Defaults to the current time.
        resolution (Optional[Tuple[int, int]]):
            Frame resolution as (width, height).
        fps (Optional[float]):
            Estimated frames per second for the device.
        frame_number (Optional[int]):
            Sequential frame counter, if available.
        device_name (Optional[str]):
            Human-readable camera or device name.
    """
    device_id: str
    frame: np.ndarray
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    resolution: Optional[Tuple[int, int]] = None  # (width, height)
    fps: Optional[float] = None
    frame_number: Optional[int] = None
    device_name: Optional[str] = None


@dataclass
class RGBFrameData(BaseFrameData):
    """Frame data for RGB cameras.

    Attributes:
        is_color (bool):
            Indicates that the frame contains 3-channel color data.
    """
    is_color: bool = True


@dataclass
class GrayFrameData(BaseFrameData):
    """Frame data for grayscale or monochrome cameras.

    Attributes:
        is_color (bool):
            Indicates that the frame contains 1-channel grayscale data.
    """
    is_color: bool = False


@dataclass
class ThermalFrameData(BaseFrameData):
    """Frame data for thermal cameras.

    Attributes:
        temperature_data (Optional[np.ndarray]):
            2D array of temperature values (same dimensions as the frame).
        temperature_unit (Optional[str]):
            Temperature unit (e.g., "C", "F").
        min_temp (Optional[float]):
            Minimum temperature detected in the frame.
        max_temp (Optional[float]):
            Maximum temperature detected in the frame.
    """

    temperature_data: Optional[np.ndarray] = None  # 2D temperature values
    temperature_unit: Optional[str] = None
    min_temp: Optional[float] = None
    max_temp: Optional[float] = None


@dataclass
class IRFrameData(BaseFrameData):
    """Frame data for infrared cameras.

    Attributes:
        heatmap_data (Optional[np.ndarray]):
            Processed heatmap visualization of IR intensity.
        raw_ir_data (Optional[np.ndarray]):
            Raw pixel intensities from the IR sensor.
    """
    heatmap_data: Optional[np.ndarray] = None
    raw_ir_data: Optional[np.ndarray] = None

@dataclass
class ErrorFrameData(BaseFrameData):
    """Frame type used to represent errors or invalid captures.

    Attributes:
        error_message (str):
            Description of the error condition. Displayed on the generated image.
    """
    error_message: str = ""

    @staticmethod
    def create(camera_id: str, message: str,
               resolution: Tuple[int, int] = (1920, 1080)) -> "ErrorFrameData":
        """Create an error frame with a black background and an embedded text message.

        Args:
            camera_id (str):
                Identifier of the camera that encountered the error.
            message (str):
                Error description to draw on the frame.
            resolution (Tuple[int, int], optional):
                Size of the generated image as (width, height). Defaults to 1920*1080.

        Returns:
            ErrorFrameData: A frame object containing the rendered error message.
        """
        # Create a black background image
        img = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

        # Draw the error message text
        cv2.putText(
            img,
            message,
            org=(50, resolution[1] // 2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255, 255, 255),
            thickness=4,
            lineType=cv2.LINE_AA,
        )

        # Return an ErrorFrameData object with this image
        return ErrorFrameData(
            device_id=camera_id,
            frame=img,
            timestamp=datetime.datetime.now(),
            resolution=resolution,
            error_message=message
        )

@dataclass
class CameraHealth:
    """Health and status information for a camera.

    Attributes:
        cam_id (str):
            Unique identifier for the camera.
        thread_id (int | None):
            Identifier of the thread running the camera logic, if applicable.
        connected (bool):
            Indicates whether the camera is currently connected and operational.
        estimated_fps (float):
            Estimated frame rate based on recent captures.
    """
    cam_id: str
    thread_id: int | None
    connected: bool
    estimated_fps: float
