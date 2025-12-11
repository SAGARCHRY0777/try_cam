
import asyncio
import psutil
import time
import threading
from typing import Dict, Optional, Callable
from devices.cameras.dataclass import BaseFrameData, ErrorFrameData
from devices.cameras.wrapper import CameraWrapper
from utils.logging_config import setup_logger
from utils.utilities import compute_config_hash, load_json

logger = setup_logger()

class CameraManager:
    """
    Manages multiple cameras with threaded capture and async supervisor/monitor loops.
    Designed to be started/stopped from a service registry that runs an asyncio event loop.

    NOTE:
      - Camera capture remains in blocking threads inside CameraWrapper.
      - Supervisor and monitor are async coroutines and must run on an asyncio event loop.
      - Blocking camera operations (stop/start/hard_update_config) are called via asyncio.to_thread()
        so the event loop is never blocked.
    """

    def __init__(self, cameras_path: str,
                 monitor_interval: float = 1.0,
                 max_stuck_time: float = 2.0,
                 notify_fn: Optional[Callable] = None,
                 eventbus = None):
        
        self.cameras_path = cameras_path
        self.camera_wrappers: Dict[str, CameraWrapper] = {}
        self._lock = threading.Lock()                    # used by camera threads and async loops
        self.monitor_interval = monitor_interval
        self.max_stuck_time = max_stuck_time
        self.notify_fn = notify_fn

        self._proc = psutil.Process()
        self._prev_thread_cpu: Dict[int, float] = {}     # tid -> cumulative CPU time
        self._last_frame_times: Dict[str, float] = {}    # cam_id -> last seen frame timestamp
        self._camera_status: Dict[str, bool] = {}        # cam_id -> connectivity

        # runtime state for async tasks
        self._is_manager_running = False
        self._task_supervisor: Optional[asyncio.Task] = None # initialised 
        self._task_monitor: Optional[asyncio.Task] = None # initialised
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # to just check it twice within the stuck timeout window 
        if self.monitor_interval >= self.max_stuck_time:
            logger.warning(
                f"monitor_interval ({self.monitor_interval}) >= max_stuck_time ({self.max_stuck_time}); "
                "supervisor may falsely restart cameras."
            )

        self._event_bus = eventbus
        self._event_bus.subscribe('camera_triggered_event', self.on_camera_trigger)

    # -----------------------------
    # Service registry style start/stop
    # -----------------------------

    # The stop is async because it has to wait for proper shutdown
    async def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Start async supervisor and monitor loops.

        Must be called when an asyncio event loop is running (either pass it in `loop`
        or call from within the running loop).
        """

        if self._is_manager_running:
            return

        logger.info("Starting CameraManager")

        # determine event loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError as e:
                logger.error("Run time error in start(): ", e)
                raise RuntimeError(
                    "No running asyncio event loop. Call start(loop) with a running loop or call from inside an event loop."
                ) from e

        self._event_loop = loop
        self._is_manager_running = True
        self.update_cameras()
        print("update done")
        
        # create tasks on the provided loop
        self._task_supervisor = loop.create_task(self._supervisor_loop(), name="camera_supervisor")
        self._task_monitor = loop.create_task(self._monitor_loop(), name="camera_monitor")

        logger.info("CameraManager started")
        self._notify("CameraManager started")

    async def stop(self, wait_for_tasks: bool = True):
        """
        Stop async loops and all cameras.

        This is an async method; it will:

          - cancel supervisor/monitor tasks
          - stop all CameraWrapper instances (using thread pool)
          - await task cancellation if wait_for_tasks=True
        """
        if not self._is_manager_running:
            return

        logger.info("Stopping CameraManager")
        self._is_manager_running = False

        # cancel tasks
        tasks = []
        if self._task_supervisor:
            self._task_supervisor.cancel()
            tasks.append(self._task_supervisor)
            self._task_supervisor = None
        if self._task_monitor:
            self._task_monitor.cancel()
            tasks.append(self._task_monitor)
            self._task_monitor = None

        # stop camera wrappers in thread pool to avoid blocking the event loop
        with self._lock: #q locking my camera wrappers to get a safe read
            wrappers = list(self.camera_wrappers.values())

        if wrappers:
            await asyncio.gather(*(asyncio.to_thread(w.stop) for w in wrappers), return_exceptions=True)

        # optionally wait for tasks to finish cancellation
        if wait_for_tasks and tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error("Error in stop():",e)
                # ignore; tasks might raise CancelledError
    
        logger.info("CameraManager stopped")
        self._notify("CameraManager stopped")

    async def on_camera_trigger(self, event):
        cam_id = event.get('cam_id')
        print("CamId is::::::::", cam_id)
        print("self.camera_wrappers:::::",self.camera_wrappers)
        if cam_id in self.camera_wrappers: 
            print("Found") 
            camera = self.camera_wrappers[cam_id]  # Gets the value (CameraWrapper object)
            print(camera)
            camera.trigger()
        else:
            print("Not found")
            logger.info(f"{cam_id} cam_id not found")


    # convenience start/stop for non-async callers - will run an event loop
    def start_sync(self, loop: Optional[asyncio.AbstractEventLoop] = None, timeout: float = 5.0):
        """
        Start manager from synchronous code.
        - If `loop` is provided, schedule coroutine on that loop (assumed to run in another thread).
        - If no loop provided and no running loop in current thread, use asyncio.run().
        - If there is a running loop in current thread, raise an informative error (caller should await instead).
        """
        if loop is None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # no loop running in this thread -> safe to create one
                asyncio.run(self.start())
                return
            # if we reach here, there *is* a loop running in this thread
            raise RuntimeError("start_sync() called from inside a running event loop — call start() (await) instead")

        # If a loop was provided, assume it's running in ANOTHER thread and use run_coroutine_threadsafe
        fut = asyncio.run_coroutine_threadsafe(self.start(), loop)
        fut.result(timeout=timeout)

    def stop_sync(self, timeout: float = 5.0):
        """
        Stop the manager from sync code. This will run the async stop() using the
        current loop if present, otherwise create a temporary event loop.
        """
        try:
            loop = asyncio.get_running_loop()
            # we are inside an event loop; schedule stop
            fut = asyncio.run_coroutine_threadsafe(self.stop(), loop)
            fut.result(timeout=timeout)
        except RuntimeError:
            # no running loop: create one temporarily
            asyncio.run(self.stop())

    # -----------------------------
    # Config file handling and updates
    # -----------------------------
    def update_cameras(self):
        """
        Synchronous update: reads config file and adds/removes/updates wrapper instances.
        Note: this interacts with CameraWrapper.start/stop which are blocking, but those
        are quick operations; if desired they can be offloaded to threads.
        """
        config = load_json(self.cameras_path,logger)
        if not config:
            logger.warning("Camera configuration is empty. No cameras to load.")
            for cam_id, wrapper in list(self.camera_wrappers.items()):
                try:
                    wrapper.stop()
                except Exception:
                    pass

            self.camera_wrappers.clear()
            self._camera_status.clear()
            self._last_frame_times.clear()
            self._prev_thread_cpu.clear()
            return

        with self._lock: # locking the camera wrappers to get a safe read
            current_ids = set(self.camera_wrappers.keys())
            new_ids = set(config.keys())

            # Add new cameras
            for cam_id in new_ids - current_ids:
                cam_cfg = config[cam_id]
                wrapper = CameraWrapper(
                    cam_id=cam_id,
                    module_path=cam_cfg["module"],
                    class_name=cam_cfg["type"],
                    config=cam_cfg["config"],
                    notify_fn=self.notify_fn,logger=logger
                )
                wrapper.start()
                self.camera_wrappers[cam_id] = wrapper
                self._last_frame_times[cam_id] = time.time()
                self._camera_status[cam_id] = wrapper.camera.is_connected()
                if wrapper.thread_id:
                    t_cpu = self._get_thread_cpu_time(wrapper.thread_id)
                    if t_cpu is not None:
                        self._prev_thread_cpu[wrapper.thread_id] = t_cpu

            # Remove deleted cameras
            for cam_id in current_ids - new_ids:
                try:
                    self.camera_wrappers[cam_id].stop()
                except Exception:
                    pass
                tid = getattr(self.camera_wrappers[cam_id], "thread_id", None)
                if tid and tid in self._prev_thread_cpu:
                    del self._prev_thread_cpu[tid]
                del self.camera_wrappers[cam_id]
                self._camera_status.pop(cam_id, None)
                self._last_frame_times.pop(cam_id, None)

            # Update existing cameras (hot updates)
            for cam_id in current_ids & new_ids:
                cam_cfg = config[cam_id]["config"]
                wrapper = self.camera_wrappers[cam_id]
                if wrapper.config_hash != compute_config_hash(cam_cfg):
                    # prefer hot update if supported
                    try:
                        wrapper.config_update(cam_cfg)
                    except AttributeError as e:
                        logger.error("Update_cameras() - Config attribute error: ",e)

    # -----------------------------
    # Access frames
    # -----------------------------
    # def get_camera_frame(self, cam_id: str) -> Optional[BaseFrameData]:
    #     with self._lock:
    #         wrapper = self.camera_wrappers.get(cam_id)
    #     if wrapper:
    #         self._last_frame_times[cam_id] = time.time()
    #         return wrapper.get_latest_frame()
    #     return ErrorFrameData.create(cam_id, "No device found")

    # def get_latest_frames(self) -> dict:
    #     """
    #     Return {cam_id: latest_frame} for all cameras.
    #     Note: does not hold the global lock for the whole iteration (reads wrappers snapshot).
    #     """
    #     with self._lock:
    #         wrappers = dict(self.camera_wrappers)
    #     frames = {}
    #     for cid, w in wrappers.items():
    #         lf = w.get_latest_frame()
    #         if lf is not None:
    #             frames[cid] = lf
    #         ## else what will happen?
    #     return frames

    # def get_buffer_frames(self, cam_id: str) -> list:
    #     with self._lock:
    #         wrapper = self.camera_wrappers.get(cam_id)
    #     if wrapper:
    #         return wrapper.get_frame_buffer()
    #     return []

    # def get_connectivity_status(self) -> dict:
    #     with self._lock:
    #         status = self._camera_status
    #     return dict(status)

    def get_health_status(self) -> dict:
        """Return all camera metrics and process info for GlobalMonitor."""
        with self._lock:
            statuses = {cid: w.get_health_status() for cid, w in self.camera_wrappers.items()}
        try:
            mem_info = self._proc.memory_info()
            mem_rss_mb = mem_info.rss / (1024 * 1024)
        except Exception:
            mem_rss_mb = -1.0
        return {
            "process": {"rss_mb": mem_rss_mb, "thread_count": self._proc.num_threads()},
            "cameras": statuses
        }

    # -----------------------------
    # Utilities
    # -----------------------------
    def _get_thread_cpu_time(self, tid: int) -> Optional[float]:
        try:
            for t in self._proc.threads():
                if t.id == tid:
                    return float(t.user_time + t.system_time)
        except Exception:
            logger.exception("Failed reading thread CPU times")
        return None

    def _notify(self, msg: str):
        """Internal notify helper (manager-level)."""
        try:
            if self.notify_fn:
                self.notify_fn("CameraManager", msg)
        except Exception:
            logger.exception("notify_fn raised in CameraManager")

    # -----------------------------
    # Async Supervisor & Monitor Loops
    # -----------------------------
    async def _supervisor_loop(self):
        """Async watchdog: restart dead or stuck camera wrappers."""
        while self._is_manager_running:
            await asyncio.sleep(self.monitor_interval)
            now = time.time()

            # snapshot wrappers under lock
            with self._lock:
                wrappers = {
                    cam_id: w for cam_id, w in self.camera_wrappers.items()
                    if hasattr(w, "thread") and w.thread is not None and w.thread.is_alive()
                }
                # wrappers = dict(self.camera_wrappers)
                last_frame_times = dict(self._last_frame_times)

            # iterate without holding lock the whole time
            for cam_id, wrapper in wrappers.items():
                try:
                    thread_alive = getattr(wrapper, "_thread", None) and wrapper._thread.is_alive()
                    if not wrapper.running or not thread_alive:
                        logger.warning(f"[Supervisor] Restarting dead camera {cam_id}")
                        
                        # run hard update in thread pool (may block), don't block event loop
                        asyncio.create_task(self._restart_wrapper_async(cam_id, wrapper, now))
                        continue

                    last_time = last_frame_times.get(cam_id, now)
                    if now - last_time > self.max_stuck_time:
                        logger.warning(f"[Supervisor] Camera {cam_id} stuck, restarting...")
                        asyncio.create_task(self._restart_wrapper_async(cam_id, wrapper, now))

                except Exception:
                    logger.exception(f"[Supervisor] Exception while supervising {cam_id}")

    async def _restart_wrapper_async(self, cam_id: str, wrapper: CameraWrapper, now: float):
        """
        Restart wrapper by calling blocking config_update_hard in thread pool.
        After restart, update bookkeeping (last_frame_times and prev thread cpu).
        """
        try:
            # perform blocking restart
            await asyncio.to_thread(wrapper.config_update_hard, wrapper.config)
        except Exception:
            logger.exception(f"[Supervisor] Exception while hard restarting {cam_id}")
            return

        # after restart, update bookkeeping under lock
        with self._lock:
            self._last_frame_times[cam_id] = now
            tid = getattr(wrapper, "thread_id", None)
            if tid:
                
                t_cpu = self._get_thread_cpu_time(tid)
                if t_cpu is not None:
                    self._prev_thread_cpu[tid] = t_cpu

### Notify_fn and _notify to be rechecked

    async def _monitor_loop(self):
        """Async monitor: collect CPU, memory, FPS, and connectivity info per camera."""
        while self._is_manager_running:
            await asyncio.sleep(self.monitor_interval)
            # now = time.time()

            # fetch process thread times once
            try:
                proc_threads = {t.id: t.user_time + t.system_time for t in self._proc.threads()}
            except Exception:
                logger.exception("Failed to read process threads from psutil")
                proc_threads = {}

            # operate on a snapshot of wrappers to avoid holding lock while doing IO/notify
            with self._lock:
                wrappers = {
                    cam_id: w for cam_id, w in self.camera_wrappers.items()
                    if hasattr(w, "_thread") and w.thread is not None and w.thread.is_alive()
                }

            for cam_id, wrapper in wrappers.items():
                try:
                    # connectivity (quick call)
                    is_conn = wrapper.camera.is_connected()
                    with self._lock:
                        self._camera_status[cam_id] = is_conn

                    # Estimate memory used by frame queue (per-camera)
                    mem_bytes = sum(frame.frame.nbytes if hasattr(frame, "frame") and frame.frame is not None else 0
                                    for frame in wrapper.frame_queue)
                    mem_mb = mem_bytes / (1024 * 1024)

                    # CPU usage for camera thread
                    tid = getattr(wrapper, "thread_id", None)
                    cpu_percent = None
                    if tid is not None:
                        prev = self._prev_thread_cpu.get(tid)
                        curr = proc_threads.get(tid)
                        if curr is not None and prev is not None:
                            delta = curr - prev
                            cpu_percent = (delta / self.monitor_interval) * 100.0
                            with self._lock:
                                self._prev_thread_cpu[tid] = curr
                        elif curr is not None and prev is None:
                            with self._lock:
                                self._prev_thread_cpu[tid] = curr
                            cpu_percent = 0.0

                    est_fps = getattr(wrapper, "estimated_fps",0.0)
                    buffered = len(wrapper.frame_queue)

                    msg = (f"[Resource] cam_id={cam_id} TID={tid} "
                        f"CPU%={'{:.2f}'.format(cpu_percent) if cpu_percent is not None else 'N/A'} "
                        f"FramesBuffered={buffered} EstimatedFPS={'{:.2f}'.format(est_fps)} "
                        f"Memory_MB={'{:.2f}'.format(mem_mb)}")
                    logger.info(msg)

                    if self.notify_fn:
                        self.notify_fn(cam_id, msg)

                    # Update last frame time for stuck detection
                    lf = wrapper.get_latest_frame()
                    if lf is not None:
                        ts = getattr(lf, "timestamp", None)
                        try:
                            import datetime
                            last_ts = ts.timestamp() if isinstance(ts, datetime.datetime) else float(ts)
                        except Exception:
                            last_ts = time.time()
                        with self._lock:
                            self._last_frame_times[cam_id] = last_ts

                except Exception as e:
                    logger.error(f"[Monitor] Exception while monitoring {cam_id}:{e}")
