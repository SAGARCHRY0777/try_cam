"""base.py

Defines an abstract base class for camera implementations, providing a
standardized interface for connecting, disconnecting, fetching frames,
streaming frames, and applying configuration updates. All concrete camera
drivers in the system should inherit from `CameraBase`.
"""

from abc import ABC, abstractmethod
from typing import Generator, Optional, Callable, Dict, Any

from devices.cameras.dataclass import BaseFrameData

class CameraBase(ABC):
    """Abstract base class for all camera implementations.

    This class defines the required interface that every camera driver must
    implement, including methods for connection management, frame capture,
    streaming, and configuration handling.
    """

    def __init__(self, cam_id: str, config: Dict[str, Any]) -> None:
        """Initialize a camera instance.

        Args:
            cam_id (str):
                Unique identifier for the camera.
            config (dict[str, Any]):
                Configuration parameters for initializing the camera.
        """
        self.camera_id: str = cam_id
        self.config: Dict[str, Any] = config
        self.connected: bool = False

    @abstractmethod
    def connect(self) -> bool:
        """Establish a connection to the camera.

        Returns:
            bool:
                True if the connection is successful; False otherwise.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect the camera safely and release hardware resources."""

    def is_connected(self) -> bool:
        """Check whether the camera is currently connected.

        Returns:
            bool:
                True if the camera is connected; False otherwise.
        """
        return self.connected

    @abstractmethod
    def fetch_frame(self) -> Optional[BaseFrameData]:
        """Fetch a single frame from the camera.

        Returns:
            Optional[BaseFrameData]:
                The captured frame object, or None if no valid frame
                could be retrieved.
        """

    def stream_frames(
        self, on_stop: Optional[Callable[[str], None]] = None
    ) -> Generator[BaseFrameData, None, None]:
        """Continuously stream frames until the camera stops providing data.

        Args:
            on_stop (Optional[Callable[[str], None]]):
                Optional callback invoked with a message when streaming stops.
                Useful for logging or error reporting.

        Yields:
            BaseFrameData:
                Frame data objects captured from the camera.
        """
        while True:
            frame_data = self.fetch_frame()
            if frame_data is not None:
                yield frame_data
            else:
                if on_stop:
                    on_stop("Camera disconnected")
                break

    def get_info(self) -> Dict[str, Any]:
        """Retrieve metadata and configuration details about the camera.

        Returns:
            dict[str, Any]:
                A dictionary containing camera ID, type, connection status,
                and configuration values.
        """
        return {
            "camera_id": self.camera_id,
            "camera_type": self.__class__.__name__,
            "connected": self.connected,
            "config": self.config,
        }

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update the camera configuration at runtime.

        Subclasses should override `_apply_config()` to push configuration
        changes to the actual hardware.

        Args:
            updates (dict[str, Any]):
                Key-value pairs of configuration parameters to update.
        """
        self.config.update(updates)
        self._apply_config(updates)

    @abstractmethod
    def _apply_config(self, updates: Dict[str, Any]) -> None:
        """Apply configuration updates to the camera hardware.

        Args:
            updates (dict[str, Any]):
                The configuration values to apply.
        """
