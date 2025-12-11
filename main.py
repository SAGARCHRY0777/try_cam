import asyncio
import cv2
from managers.event_bus_manager import EventBus
from managers.camera import CameraManager

event = EventBus()

def notify(cam_id, msg):
    print(f"[{cam_id}] {msg}")

async def main():
    cameras_path = "cameras.json"
    cam_mgr = CameraManager(
        cameras_path=cameras_path,
        monitor_interval=5,
        max_stuck_time=10,
        notify_fn=notify,
        eventbus=event
    ) # create the camera manager

    loop = asyncio.get_running_loop() #get the currently active event loop
    await cam_mgr.start(loop=loop) # start the camera manager

    print("[Test] Press 'C' to trigger camera, 'Q' to quit.")

    try:
        while True:
            # cam_mgr.update_cameras()

            # Display frames
            for cam_id, wrapper in cam_mgr.camera_wrappers.items():
                frame_data = wrapper.get_latest_frame()
                # print(frame_data)
                if frame_data is not None and hasattr(frame_data, "frame"):
                    cv2.imshow(f"Camera {cam_id}", frame_data.frame)

            key = cv2.waitKey(1) & 0xFF

            for cam_id, wrapper in cam_mgr.camera_wrappers.items():
                frames = wrapper.get_trigger_buffer()
                if not frames:
                    continue
                for idx, frame in enumerate(frames):
                    frame = frame.frame
                    if frame is not None:
                        window_name = f"Camera {cam_id} Frame {idx}"
                        cv2.imshow(window_name, frame)
            wrapper.trigger_buffer.clear()
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            # 🔹 Trigger camera only on pressing 'C'
            if key == ord('c'):
                cam_id = "cam01"
                print(f"[KEY PRESS] Triggering {cam_id}")
                await event.publish('camera_triggered_event', {'cam_id': cam_id})

            # 🔹 Quit if 'Q' pressed
            if key == ord('q'):
                break

            await asyncio.sleep(0.01)

    finally:
        print("[Test] Stopping CameraManager...")
        for wrapper in cam_mgr.camera_wrappers.values():
            wrapper.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main()) # event loop is created and passed to the main function #thread local storage to store this
