from picamera2 import Picamera2
from libcamera import Transform
from tracker_display import TrackerDisplay
# from tracker import TrackerHeadless as Tracker  # swap here for headless

def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        controls={"FrameRate": 120},
        transform=Transform(hflip=True, vflip=True)
    ))
    picam2.start()
    picam2.set_controls({"AfMode": 2})  # continuous autofocus

    tracker = TrackerDisplay()

    try:
        while not tracker.should_stop():
            frame = picam2.capture_array()
            error_x, error_y, state = tracker.update(frame)

            # TODO: coords_to_rads(error_x, error_y)
            # TODO: send_to_mcu(rads)

    finally:
        tracker.stop()
        picam2.stop()


if __name__ == "__main__":
    main()
