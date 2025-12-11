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
import os
import cv2
import traceback

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
        
        # Separate buffer for triggered frames (stores ONLY triggered frames)
        self.trigger_buffer_size = config.get("frames_to_fetch", 5)
        self.trigger_buffer: deque[BaseFrameData] = deque(maxlen=self.trigger_buffer_size)
        
        self.latest_frame: Optional[BaseFrameData] = None

        # --- Threading state ---
        self.running: bool = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.thread_id: Optional[int] = None

        # --- Metrics ---
        self._last_frame_timestamp: Optional[float] = None
        self.estimated_fps: float = config.get("fps", 0.0)

    # def _get_capture_mode(self):
    #     """Get capture mode with backward compatibility for 'trigger' boolean.
        
    #     Returns:
    #         str: One of 'continuous', 'trigger', or 'hybrid'
            
    #     Notes:
    #         - Checks for new 'capture_mode' parameter first
    #         - Falls back to old 'trigger' boolean for backward compatibility
    #         - Defaults to 'continuous' if neither is specified
    #     """
    #     # Check for new 'capture_mode' parameter first
    #     if "capture_mode" in self.config:
    #         mode = self.config.get("capture_mode")
    #         if mode in ["continuous", "trigger", "hybrid"]:
    #             return mode
    #         else:
    #             self.logger.warning(
    #                 f"[{self.cam_id}] Invalid capture_mode '{mode}', defaulting to continuous"
    #             )
    #             return "continuous"
    #     return "continuous"

    def _get_capture_mode(self) -> str:
        """Get capture mode with backward compatibility and comprehensive error handling.
        
        Returns:
            str: One of 'continuous', 'trigger', or 'hybrid'
        
        Notes:
            - Prioritizes 'capture_mode' key
            - Falls back to 'trigger' boolean for backward compatibility  
            - Defaults to 'continuous' for safety
            - Logs all decisions and validation failures
        """
        try:
            # Check for new 'capture_mode' parameter first (prioritized)
            if "capture_mode" in self.config:
                mode = self.config.get("capture_mode", "").strip().lower()
                self.logger.info(f"[{self.cam_id}] capture_mode found in config: '{mode}'")
                
                if mode in ["continuous", "trigger", "hybrid"]:
                    self.logger.info(f"[{self.cam_id}] Valid capture_mode: '{mode}'")
                    return mode
                else:
                    self.logger.warning(
                        f"[{self.cam_id}] Invalid capture_mode '{mode}', "
                        f"valid options: ['continuous', 'trigger', 'hybrid']. Defaulting to 'continuous'"
                    )
                    return "continuous"
            # Default case
            else:
                self.logger.info(f"[{self.cam_id}] No capture_mode or trigger specified, defaulting to 'continuous'")
                return "continuous"
        
        except Exception as e:
            self.logger.error(
                f"[{self.cam_id}] Error determining capture_mode: {e}, "
                f"config keys: {list(self.config.keys())}, defaulting to 'continuous'"
            )
            self.logger.debug(f"[{self.cam_id}] Full traceback: {traceback.format_exc()}")
            return "continuous"


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
            module = importlib.import_module(module_path) #import a module dynamically 
            cls = getattr(module, class_name)
            print("cls-------",cls)
            if not issubclass(cls, CameraBase):#return true and false 
                #### Raise should be handled in a better way
                self.logger.error(f"{class_name} must subclass CameraBase")
                raise TypeError(f"{class_name} must subclass CameraBase")
            return cls
        except Exception as e:
            self.logger.error(f"Camera class load failed: {module_path}.{class_name} - {e}")
            raise

    def start(self):
        """Start the camera with proper handling for trigger or continuous mode."""

        # Event to sync thread startup
        # synv for the main thread and the camera thread 
        # main thread --- wrapper .start--- starts camera thread --- connecting camera 
        self.started_event = threading.Event()  # event to have a synchronization between the main thread and the camera thread
        self.started_event.clear() #reset the event to false # need the thread to wait for the event to be set

        # Thread target decides connection + capture type
        def thread_target():
            # Connect inside the thread to avoid main thread blocking
            capture_mode = self._get_capture_mode()  # get the capture mode from the config
            print(f"[{self.cam_id}] Starting in {capture_mode.upper()} mode")

            if capture_mode == "continuous":
                try:
                    self._run_continuous()
                except Exception as e:
                    self.logger.exception(f"[{self.cam_id}] Error in continuous loop: {e}")

            elif capture_mode == "trigger":
                try:
                    self._run_trigger()
                except asyncio.CancelledError:
                    self.logger.exception(f"[{self.cam_id}] Trigger task cancelled")
                except Exception as e:
                    self.logger.exception(f"[{self.cam_id}] Error in trigger loop: {e}")

            elif capture_mode == "hybrid":
                # Hybrid mode: both continuous and trigger
                try:
                    self._run_hybrid_loop()
                except Exception as e:
                    self.logger.exception(f"[{self.cam_id}] Error in hybrid loop: {e}")

            else:
                self.logger.error(f"[{self.cam_id}] Unknown capture_mode: {capture_mode}")

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

    def _run_continuous(self):
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

    def _run_trigger(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._trigger_event = asyncio.Event()
        self._loop.create_task(self._run_trigger_based_fetch())

        print(f"[{self.cam_id}] Event loop started in thread {threading.get_ident()}")
        self._loop.run_forever()

    async def _run_trigger_based_fetch(self):
        print(f"Entered trigger-based fetch for camera {self.cam_id}...")

        if not self.camera.connect():
            self.running = False
            self.logger.error(f"Camera {self.cam_id} connection failed")
            self._notify(f"Camera {self.cam_id} connection failed")
            self.started_event.set()  # notify main thread
            return

        # Connected successfully
        self.running = True
        self.started_event.set()  # notify main thread

        # Ensure trigger_buffer always holds latest frames based on config
        frames_to_fetch = self.config.get("frames_to_fetch", 20)
        self.trigger_buffer = deque(maxlen=frames_to_fetch)

        frame = self.camera.fetch_frame()
        self._process_frame(frame)
        self.trigger_buffer.append(frame)
        while self.running:
            try:
                # 🔹 Wait here doing nothing until trigger event
                print(f"[{self.cam_id}] Waiting for trigger...")
                await self._trigger_event.wait()
                print(f"[{self.cam_id}] Trigger detected!")

                trigger_start = time.time()
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                # Clear previous frames (if any)
                with self._lock:
                    self.trigger_buffer.clear()

                saved = 0

                # 🔹 After trigger → capture N frames
                for i in range(frames_to_fetch):
                    frame = self.camera.fetch_frame()

                    if frame is not None:
                        self._process_frame(frame)
                        with self._lock:
                            self.trigger_buffer.append(frame)
                        saved += 1

                total_time = time.time() - trigger_start
                self._trigger_event.clear()

                print(f"[{self.cam_id}] ═══ TRIGGER COMPLETE ═══")
                print(f"[{self.cam_id}] Captured {saved}/{frames_to_fetch} frames in {total_time:.3f}s")
                print(f"[{self.cam_id}] Trigger buffer size: {len(self.trigger_buffer)}")

            except Exception as e:
                self._handle_error(e)

            await asyncio.sleep(0.001)  # light yield to event-loop


    # def _run_hybrid_loop(self):
    #     """Run both continuous and trigger-based capture simultaneously"""
        
    #     self._loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(self._loop)
        
    #     # Connect camera
    #     if not self.camera.connect():
    #         self.running = False
    #         self.started_event.set()
    #         return
        
    #     self.running = True
    #     self.started_event.set()
        
    #     # Create trigger event
    #     self._trigger_event = asyncio.Event()
        
    #     # Schedule BOTH tasks
    #     continuous_task = self._loop.create_task(self._continuous_background_task())
    #     trigger_task = self._loop.create_task(self._trigger_capture_task())
        
    #     print(f"[{self.cam_id}] Hybrid mode: Event loop started with 2 tasks")
    #     # Run event loop
    #     self._loop.run_forever()

    # async def _continuous_background_task(self):
    #     """Continuously fetch frames in the background."""
    #     print(f"[{self.cam_id}] Continuous background task started")
    #     while self.running:
    #         try:
    #             # Run fetch_frame in a separate thread to avoid blocking the event loop
    #             frame = await asyncio.to_thread(self.camera.fetch_frame)
                
    #             # Process frame (stores in frame_queue only)
    #             self._process_frame(frame)
                
    #         except Exception as e:
    #             print(f"[{self.cam_id}] Continuous task error: {e}")
            
    #         # Yield control to allow trigger task to run
    #         await asyncio.sleep(0.01)

    # async def _trigger_capture_task(self):
    #     """Wait for trigger events and capture frames from the FRAME QUEUE."""
    #     print(f"[{self.cam_id}] Trigger capture task started (using frame_queue)")
        
    #     # Create fetched_frames directory if it doesn't exist
    #     # Use absolute path relative to wrapper.py location
    #     save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "fetched_frames")
    #     save_dir = os.path.abspath(save_dir)
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     # Log save directory for debugging
    #     print(f"[{self.cam_id}] Frames will be saved to: {save_dir}")
    #     self.logger.info(f"[{self.cam_id}] Save directory: {save_dir}")
        
    #     while self.running:
    #         try:
    #             # Wait for trigger event
    #             print(f"[{self.cam_id}] Waiting for trigger event...")
                
    #             # Polling fallback for debugging
    #             while not self._trigger_event.is_set():
    #                 await asyncio.sleep(0.1)
    #             trigger_start_time = time.time()
    #             print(f"[{self.cam_id}] ═══ TRIGGER RECEIVED at {trigger_start_time} ═══")
                
    #             frames_to_fetch = self.config.get("frames_to_fetch", 1)
    #             print(f"[{self.cam_id}] Capturing {frames_to_fetch} frames from frame_queue...")
                
    #             # Create timestamp for this trigger burst
    #             trigger_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
                
    #             # Get frames and update trigger_buffer in single lock operation
    #             with self._lock:
    #                 # Get current buffer info
    #                 old_buffer_size = len(self.trigger_buffer)
    #                 print(f"[{self.cam_id}] 📊 frame_queue size: {len(self.frame_queue)}")
    #                 if old_buffer_size > 0:
    #                     print(f"[{self.cam_id}] 🗑️ Clearing {old_buffer_size} old frames from trigger_buffer...")
                    
    #                 # Get frames and update buffer
    #                 frames = list(self.frame_queue)[-frames_to_fetch:]
    #                 self.trigger_buffer.extend(frames)
                    
    #                 new_buffer_size = len(self.trigger_buffer)
    #                 print(f"[{self.cam_id}] ✅ trigger_buffer updated: {old_buffer_size} → {new_buffer_size} frames")
                
    #             print(f"[{self.cam_id}] Retrieved {len(frames)} frames from frame_queue (requested {frames_to_fetch})")
                
    #             # Debug: Print frame details
    #             if frames:
    #                 first_frame = frames[0]
    #                 last_frame = frames[-1]
    #                 print(f"[{self.cam_id}] First frame timestamp: {first_frame.timestamp if hasattr(first_frame, 'timestamp') else 'N/A'}")
    #                 print(f"[{self.cam_id}] Last frame timestamp: {last_frame.timestamp if hasattr(last_frame, 'timestamp') else 'N/A'}")
                
    #             # Save frames to disk
    #             save_start_time = time.time()
    #             saved_count = 0
                
    #             for i, frame in enumerate(frames, 1):
    #                 if frame is not None and hasattr(frame, 'frame') and frame.frame is not None:
    #                     filename = f"{self.cam_id}_trigger_{trigger_timestamp}_frame{i:03d}.jpg"
    #                     filepath = os.path.join(save_dir, filename)
                        
    #                     # Use to_thread for disk I/O to avoid blocking
    #                     success = await asyncio.to_thread(cv2.imwrite, filepath, frame.frame)
                        
    #                     if success:
    #                         print(f"[{self.cam_id}] ✅ Saved: {filename}")
    #                         saved_count += 1
    #                     else:
    #                         print(f"[{self.cam_id}] ❌ Failed to save: {filename}")
    #                 else:
    #                     print(f"[{self.cam_id}] ⚠️ Frame {i} invalid, cannot save")
                
    #             save_end_time = time.time()
    #             total_time = save_end_time - trigger_start_time
    #             save_time = save_end_time - save_start_time
                
    #             # Reset event
    #             self._trigger_event.clear()
                
    #             print(f"[{self.cam_id}] ═══ TRIGGER COMPLETE ═══")
    #             print(f"[{self.cam_id}] Saved {saved_count}/{len(frames)} frames to disk in {save_time:.3f}s (total: {total_time:.3f}s)")
    #             print(f"[{self.cam_id}] Waiting for next trigger...")
                
    #         except Exception as e:
    #             print(f"[{self.cam_id}] Trigger task error: {e}")
    #             import traceback
    #             traceback.print_exc()
    #             await asyncio.sleep(0.1)


    def _run_hybrid_loop(self):
        """Run both continuous and trigger-based capture simultaneously"""
        
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        # Connect camera
        if not self.camera.connect():
            self.running = False
            self.started_event.set()
            return
        
        self.running = True
        self.started_event.set()
        
        # Create trigger event
        self._trigger_event = asyncio.Event()
        
        # Schedule both tasks
        self._loop.create_task(self._continuous_background_task())
        self._loop.create_task(self._trigger_capture_task())
        
        print(f"[{self.cam_id}] Hybrid mode: Event loop started with 2 tasks")
        self._loop.run_forever()


    async def _continuous_background_task(self):
        """Continuously fetch frames in the background and append to frame_queue"""
        print(f"[{self.cam_id}] Continuous background task started")
        
        while self.running:
            try:
                # Fetch frame without blocking event loop
                frame = await asyncio.to_thread(self.camera.fetch_frame)
                if frame is not None:
                    self._process_frame(frame)  # updates frame_queue internally
            except Exception as e:
                print(f"[{self.cam_id}] Continuous task error: {e}")
            await asyncio.sleep(0.005)  # small yield to allow trigger task to run


    async def _trigger_capture_task(self):
        """Capture frames on trigger: pre + post frames from existing frame_queue only"""
        print(f"[{self.cam_id}] Trigger capture task started")
        
        pre_trigger_count = self.config.get("pre_trigger_frames", 5)
        post_trigger_count = self.config.get("frames_to_fetch", 50)
        
        while self.running:
            try:
                # Wait for trigger
                print(f"[{self.cam_id}] Waiting for trigger event...")
                await self._trigger_event.wait()
                trigger_start_time = time.time()
                
                with self._lock:
                    self.trigger_buffer.clear()
                    # Copy last X pre-trigger frames
                    pre_frames = list(self.frame_queue)[-pre_trigger_count:]
                    self.trigger_buffer.extend(pre_frames)
                print(f"[{self.cam_id}] Copied {len(pre_frames)} pre-trigger frames")
                
                # Wait until post-trigger frames appear in frame_queue
                post_frames_copied = 0
                while post_frames_copied < post_trigger_count:
                    await asyncio.sleep(0.005)  # yield to continuous fetch
                    with self._lock:
                        # Take only frames that arrived after trigger
                        total_frames = list(self.frame_queue)
                        # Exclude pre-trigger frames from counting
                        post_frames = total_frames[-post_trigger_count:]
                        self.trigger_buffer = pre_frames + post_frames
                        post_frames_copied = len(post_frames)
                
                total_time = time.time() - trigger_start_time
                print(f"[{self.cam_id}] Trigger complete: {len(self.trigger_buffer)} frames captured in {total_time:.3f}s")
                
                # Reset trigger event
                self._trigger_event.clear()
                
            except Exception as e:
                print(f"[{self.cam_id}] Trigger task error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(0.05)

    def trigger(self):
        """Called by CameraManager when this camera is triggered"""
        print("Entered Trigger.......")
        
        if self._loop and self._trigger_event:
            print(f"Setting event for the cam01. Event ID: {id(self._trigger_event)}")
            
            def set_event_debug():
                print(f"[{self.cam_id}] executing set() in loop. Event ID: {id(self._trigger_event)}")
                self._trigger_event.set()
                print(f"[{self.cam_id}] Event set status: {self._trigger_event.is_set()}")
            
            self._loop.call_soon_threadsafe(set_event_debug)
        else:
            print(f"Cannot trigger: loop={self._loop}, event={self._trigger_event}")

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
    
    def get_trigger_buffer(self) -> list:
        """Retrieve triggered frames from the trigger buffer.
        
        Returns:
            list[BaseFrameData]: Triggered frames (populated only after trigger events).
        
        Example:
            wrapper = camera_manager.camera_wrappers['cam01']
            triggered_frames = wrapper.get_trigger_buffer()
            if triggered_frames:
                latest = triggered_frames[-1]
                print(f"Timestamp: {latest.timestamp}")
        """
        with self._lock:
            return list(self.trigger_buffer)

    # locking the snapshot of camera health
    # returns list[BaseFrameData]
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
