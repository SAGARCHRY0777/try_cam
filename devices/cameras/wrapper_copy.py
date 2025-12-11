"""wrapper.py

Provides the `CameraWrapper` class, which encapsulates a camera driver,
its capture thread, buffering logic, configuration management, and runtime
health reporting. The wrapper abstracts dynamic camera loading, frame
processing, FPS estimation, and safe error handling.
"""
import asyncio
import importlib
import datetime
import time
import json
import threading
from collections import deque
from typing import Optional, Callable

from devices.cameras.base import CameraBase
from devices.cameras.dataclass import BaseFrameData, ErrorFrameData, CameraHealth
from utils.utilities import compute_config_hash

class CameraWrapper:
    """Wraps a camera implementation with threading, buffering and config logic.

    Responsibilities:
        • Dynamically load the camera class from a module.
        • Manage a dedicated capture thread per camera.
        • Store recent frames using a deque buffer.
        • Track performance metrics (FPS, timestamps).
        • Provide safe restart and configuration update mechanisms.
        • Generate `ErrorFrameData` for invalid frames.
    """
    def __init__(self, cam_id: str, module_path: str, class_name: str,
                 config: dict, logger: callable, notify_fn: Optional[Callable] = None):
        """Initialize the camera wrapper and prepare the camera instance.

        Args:
            cam_id (str):
                Unique identifier for the camera.
            module_path (str):
                Python module path where the camera class is located.
            class_name (str):
                Name of the camera class to load.
            config (dict):
                Configuration dictionary for the camera.
            logger (Callable):
                Logger function or object for debug/info/error output.
            notify_fn (Callable | None):
                Optional callback invoked on camera lifecycle events.
        """
        self.cam_id = cam_id
        self.module_path = module_path
        self.class_name = class_name
        self.config = config
        self.config_hash = compute_config_hash(config)
        self.notify_fn = notify_fn
        self.logger = logger

        # --- Camera instance ---
        cameraclass = self.dynamic_camera_module_import(self.module_path, self.class_name)
        self.camera: CameraBase = cameraclass(self.cam_id,self.config)

        # --- Frame buffer ---
        self.buffer_size = config.get("buffer_size", 30)
        self.frame_queue: deque[BaseFrameData] = deque(maxlen=self.buffer_size)
        self.latest_frame: Optional[BaseFrameData] = None

        # --- Threading state ---
        self.running: bool = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.thread_id: Optional[int] = None

        # --- Metrics ---
        self._last_frame_timestamp: Optional[float] = None
        self.estimated_fps: float = config.get("fps", 0.0)

    def dynamic_camera_module_import(self,module_path: str, class_name: str):
        """Dynamically import a camera class from the specified module.

        Args:
            module_path (str):
                Path to the module containing the camera class.
            class_name (str):
                Name of the camera class to load.

        Returns:
            type:
                The camera class object.

        Raises:
            TypeError: If the loaded class does not inherit from `CameraBase`.
            Exception: If import or lookup fails for any reason.
        """
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            if not issubclass(cls, CameraBase):
                #### Raise should be handled in a better way
                self.logger.error(f"{class_name} must subclass CameraBase")
                raise TypeError(f"{class_name} must subclass CameraBase")
            return cls
        except Exception as e:
            self.logger.error(f"Camera class load failed: {module_path}.{class_name} - {e}")
            raise

    # def start(self):
    #     """Start the camera and launch its capture thread.

    #     Notes:
    #         Camera connection may block if many cameras are being initialized.
    #     """
    #     #### The camera.connect may cause startup delay when lots of cameras are connected.
    #     #### Might have to move this inside the thread part
    #     if not self.camera.connect():
    #         self.logger.error(f"Initial connection failed for {self.cam_id}")
    #         self._notify(f"Camera {self.cam_id} connection failed")
    #         return

    #     self._thread = threading.Thread(target=self._run,
    #                                     name=self.config.get("name",f"Camera-{self.cam_id}"))
    #     self._thread.start()

    #     # To get thread ID, keep checking till the thread is created.
    #     while self._thread.ident is None:
    #         time.sleep(0.001)
    #     self.thread_id = self._thread.ident
    #     self.running = True

    #     self.logger.info(f"Camera thread started: {self.cam_id} (TID={self.thread_id})")
    #     self._notify(f"Camera {self.cam_id} started")

    # def _run(self):
    #     """Main camera capture loop executed inside a dedicated thread.

    #     Continuously fetches frames, processes them, and handles exceptions.
    #     """
    #     #warm up fetch
    #     frame = self.camera.fetch_frame()

    #     #continuous fetch
    #     while self.running:
    #         try:
    #             frame = self.camera.fetch_frame()
    #             self._process_frame(frame)
    #         except Exception as e:
    #             self._handle_error(e)
    #         ### The time.sleep to be rechecked for large number of cameras
    #         time.sleep(0.01)

    # def stop(self):
    #     """Stop the camera capture thread and disconnect the device.

    #     Ensures thread termination and safe disconnection with timeout fallback.
    #     """
    #     self.running = False
    #     if self._thread and self._thread.is_alive():
    #         # after timeout, if the join is not successful,
    #         # the resources may still be held by the device. Need to handle this
    #         self._thread.join(timeout=2)
    #     try:
    #         self.camera.disconnect()
    #         self.logger.info(f"Camera stopped: {self.cam_id}")
    #         self._notify(f"Camera {self.cam_id} stopped")
    #     except Exception as e:
    #         self.logger.info(f"Camera Exception: {self.cam_id}, {e}")
    #         self._notify(f"Camera {self.cam_id} exception {e}")


    ##################################### new change: connection pushed inside threading

    # def start(self):
    #     """Start the camera in a dedicated capture thread."""

    #     # Event for sync between thread and main thread
    #     self.started_event = threading.Event()
    #     self.started_event.clear()

    #     self._thread = threading.Thread(
    #         target=self._run,
    #         name=self.config.get("name", f"Camera-{self.cam_id}")
    #     )
    #     self._thread.start()

    #     # Wait for thread to finish connect() or fail
    #     if not self.started_event.wait(timeout=3):
    #         self.logger.error(f"Camera {self.cam_id} startup timeout")
    #         return

    #     # Thread may have exited early due to connection failure
    #     if not self.running:
    #         self.logger.error(f"Camera {self.cam_id} failed to start")
    #         return

    #     self.thread_id = self._thread.ident
    #     self.logger.info(f"Camera thread started: {self.cam_id} (TID={self.thread_id})")
    #     self._notify(f"Camera {self.cam_id} started")


    def start(self):
        """Start the camera with proper handling for trigger or continuous mode."""

        # Event to sync thread startup
        self.started_event = threading.Event()
        self.started_event.clear()

        # Thread target decides connection + capture type
        def thread_target():
            # Connect inside the thread to avoid main thread blocking
            

            if self.config.get("trigger"):
                # Trigger-based loop
                try:
                    self._run_trigger_loop()
                except asyncio.CancelledError:
                    self.logger.exception(f"[{self.cam_id}] Trigger task cancelled")
                except Exception as e:
                    self.logger.exception(f"[{self.cam_id}] Error in trigger loop: {e}")
            else:
                # Continuous fetch loop
                try:
                    self._run()
                except Exception as e:
                    self.logger.exception(f"[{self.cam_id}] Error in continuous loop: {e}")

        # Start the thread
        self._thread = threading.Thread(
            target=thread_target,
            name=self.config.get("name", f"Camera-{self.cam_id}")
        )
        self._thread.start()

        # Wait for thread to complete connect() or fail
        if not self.started_event.wait(timeout=3):
            self.logger.error(f"Camera {self.cam_id} startup timeout")
            return

        # Check if thread started properly
        if not self.running:
            self.logger.error(f"Camera {self.cam_id} failed to start")
            return

        # Get thread ID
        while self._thread.ident is None:
            time.sleep(0.001)
        self.thread_id = self._thread.ident

        self.logger.info(f"Camera thread started: {self.cam_id} (TID={self.thread_id})")
        self._notify(f"Camera {self.cam_id} started")

    def _run_trigger_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._trigger_event = asyncio.Event()
        self._loop.create_task(self._run_trigger_based_fetch())

        print(f"[{self.cam_id}] Event loop started in thread {threading.get_ident()}")
        self._loop.run_forever()

    def trigger(self):
        """Called by CameraManager when this camera is triggered"""
        print("Entered Trigger.......")
        
        if self._loop and self._trigger_event:
            print("Setting event for the cam01")
            # self._trigger_event.set()
            self._loop.call_soon_threadsafe(self._trigger_event.set)

    async def _run_trigger_based_fetch(self):
        print(f"Entered trigger-based fetch for camera {self.cam_id}...")
        if self._trigger_event is None:
            print("Yes its none")
        else:
            print("No its not None")
        
        if not self.camera.connect():
                self.running = False
                self.logger.error(f"Camera {self.cam_id} connection failed")
                self._notify(f"Camera {self.cam_id} connection failed")
                self.started_event.set()  # notify main thread
                return

        # Connected successfully
        self.running = True
        self.started_event.set()  # notify main thread
        while self.running:
            frame = self.camera.fetch_frame()
            self._process_frame(frame)
            try:
                # Wait for trigger event
                print("Entered and waitin for event to set........")
                await self._trigger_event.wait()
                print("Waiting completed...........")
                # Clear the event immediately so it can be triggered again
                
                print(f"Camera {self.cam_id} triggered, fetching frames...")
                for i in range(self.config["frames_to_fetch"]):
                    print(i)
                    frame = self.camera.fetch_frame()
                    self._process_frame(frame)
                
                self._trigger_event.clear()
            except Exception as e:
                self._handle_error(e)
            ### The time.sleep to be rechecked for large number of cameras
            await asyncio.sleep(0.01)

    def _run(self):
        """Camera loop running inside thread."""

        # Perform connection inside thread
        if not self.camera.connect():
            self.running = False
            self.started_event.set()   # signal startup failure
            return

        # Connected successfully
        self.running = True
        self.started_event.set()

        # Warm-up frame
        try:
            frame = self.camera.fetch_frame()
            self._process_frame(frame)
        except Exception as e:
            self._handle_error(e)

        # Continuous loop
        while self.running:
            try:
                frame = self.camera.fetch_frame()
                self._process_frame(frame)
            except Exception as e:
                self._handle_error(e)

            time.sleep(0.01)

    def stop(self):
        """Stop the capture thread and disconnect camera."""
        self.running = False

        # Stop trigger-mode event loop if running
        if hasattr(self, "_loop") and self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

        try:
            self.camera.disconnect()
            self.logger.info(f"Camera stopped: {self.cam_id}")
            self._notify(f"Camera {self.cam_id} stopped")
        except Exception as e:
            self.logger.error(f"Camera {self.cam_id} disconnection error: {e}")
            self._notify(f"Camera {self.cam_id} exception {e}")

    def config_update(self, updates: dict):
        """Apply a soft configuration update to the camera.

        Args:
            updates (dict):
                Key-value pairs of configuration parameters to update.

        Notes:
            If update fails, falls back to a full restart via `config_update_hard()`.
        """
        changed = {k: v for k, v in updates.items() if self.config.get(k) != v}
        if changed:
            try:
                self.camera.update_config(changed)
                self.config.update(changed)
                self.config_hash = compute_config_hash(self.config)
                if "buffer_size" in changed:
                    self.buffer_size = self.config.get("buffer_size", 30)
                    self.frame_queue = deque(self.frame_queue, maxlen=self.buffer_size)
                self.logger.info(f"config applied for {self.cam_id}: {changed}")
                self._notify(f"config applied for {self.cam_id}")
            except Exception as e:
                self.logger.error(f"Failed soft config for {self.cam_id}: {e}")
                self._notify(f"Failed soft config for{self.cam_id}: {e}")
                self.config_update_hard(updates)

    def config_update_hard(self, new_config: dict):
        """Apply a complete configuration update by restarting the camera.

        Args:
            new_config (dict | str):
                New configuration dictionary or JSON string.
        """
        if isinstance(new_config, str):
            new_config = json.loads(new_config)
        self.logger.info(f"Hard update (restart) for {self.cam_id}")
        self.stop()
        self.config = new_config
        self.config_hash = compute_config_hash(new_config)
        self.buffer_size = new_config.get("buffer_size", 30)
        self.frame_queue = deque(maxlen=self.buffer_size)
        cameraclass = self.dynamic_camera_module_import(self.module_path, self.class_name)
        try:
            self.camera = cameraclass(self.cam_id,self.config)
            self.start()
        except Exception as e:
            self.logger.error(f"Failed hard config for {self.cam_id}: {e}")
            self._notify(f"Failed hard config for{self.cam_id}: {e}")

    def get_latest_frame(self) -> Optional[BaseFrameData]:
        """Get the most recently captured frame.

        Returns:
            Optional[BaseFrameData]: Latest frame, or None if unavailable.
        """
        with self._lock:
            return self.latest_frame

    def get_frame_buffer(self) -> list:
        """Retrieve all frames stored in the buffer.

        Returns:
            list[BaseFrameData]: A list of recent frames.
        """
        with self._lock:
            return list(self.frame_queue)

    def get_health_status(self) -> dict:
        """Retrieve the current health and status metrics for the camera.

        Returns:
            CameraHealth: Encapsulated health information including
            thread ID, FPS, and connection status.
        """
        with self._lock:
            return CameraHealth(
                cam_id=self.cam_id,
                thread_id=self.thread_id,
                connected=self.camera.is_connected() if self.camera else False,
                estimated_fps=self.estimated_fps
            )

    def _notify(self, message: str):
        """Invoke the notification callback if provided.

        Args:
            message (str): Notification message.

        Notes:
            Exceptions are caught and logged to avoid breaking camera logic.
        """
        try:
            if self.notify_fn:
                self.notify_fn(self.cam_id, message)
        except Exception:
            self.logger.exception("Notification function raised an exception")

    def _process_frame(self, frame: Optional[BaseFrameData]):
        """Process a captured frame and update internal state.

        Args:
            frame (Optional[BaseFrameData]):
                Captured frame or None if unavailable.
        """
        with self._lock:
            if frame is None:
                self.latest_frame = ErrorFrameData.create(self.cam_id, "No Frame")
                #### Comment return and verify the behaviour
                return

            ts = self._extract_timestamp(frame)
            self._update_fps(ts)
            self._last_frame_timestamp = ts

            self.latest_frame = frame
            self.frame_queue.append(frame)

    def _extract_timestamp(self, frame) -> float:
        """Extract a numeric timestamp from the frame.

        Args:
            frame (BaseFrameData):
                Frame containing optional timestamp.

        Returns:
            float: UNIX timestamp (fallback uses current time).
        """
        ts = time.time()
        now_ts = getattr(frame, "timestamp", None)

        if now_ts is not None:
            try:
                if isinstance(now_ts, datetime.datetime):
                    ts = now_ts.timestamp()
                else:
                    ts = float(now_ts)
            except Exception:
                ts = time.time()

        return ts

    def _update_fps(self, ts: float):
        """Update FPS estimation using an exponential moving average.

        Args:
            ts (float): Timestamp of the current frame.
        """
        if self._last_frame_timestamp:
            delta = ts - self._last_frame_timestamp
            if delta > 0:
                fps = 1.0 / delta
                self.estimated_fps = (
                    0.9 * self.estimated_fps + 0.1 * fps
                    if self.estimated_fps else fps
                )

    def _handle_error(self, error: Exception):
        """Handle exceptions arising during frame capture.

        Generates an error frame, logs the issue, and attempts to disconnect
        the camera to unblock hardware operations.

        Args:
            error (Exception):
                Exception raised during frame fetching.
        """
        with self._lock:
            self.latest_frame = ErrorFrameData.create(self.cam_id, "Device Error")

        try:
            self.camera.disconnect()  # force unblocking if needed
        except Exception:
            pass

        self.logger.error(f"Error fetching frame from {self.cam_id}: {error}")
