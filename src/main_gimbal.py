import re
import math
import time
import threading
import serial
import serial.tools.list_ports
from picamera2 import Picamera2
from libcamera import Transform
from tracker_display import TrackerDisplay
from tracker import TrackState

# ── Camera / optics ───────────────────────────────────────────────────────────
FOCAL_LENGTH_PX  = 940    # imx708 (Camera Module 3), 1280x720
TARGET_HEIGHT_MM = 100    # assumed target height for distance estimation

# ── Mount geometry ────────────────────────────────────────────────────────────
PAN_OFFSET_MM    = 63.5   # pan axis -> tilt axis centre
TILT_OFFSET_MM   = 30.0   # tilt axis -> camera sensor

# ── Control ───────────────────────────────────────────────────────────────────
K_P_PAN          = 1.2
K_P_TILT         = 0.6
K_I_PAN          = 0.0    # tune after P is stable
K_I_TILT         = 0.0
K_D_PAN          = 0.1    # tune after P+I is stable
K_D_TILT         = 0.02
EMA_ALPHA        = 0.6    # only affects D term — 0=max smoothing, 1=no smoothing
INTEGRAL_CLAMP   = 0.5    # radians — prevents windup
DEAD_ZONE_PX     = 3
SEND_INTERVAL    = 0.05   # 20 Hz

# ── Serial ────────────────────────────────────────────────────────────────────
BAUD_RATE        = 115200
_ACTUAL_RE       = re.compile(r'AT([+-]?\d+\.\d+)P([+-]?\d+\.\d+)')


# ── Geometry ──────────────────────────────────────────────────────────────────

def pixel_to_angles(error_x, error_y):
    """Pixel offset from frame centre -> camera-frame angular offset (radians)."""
    return math.atan2(error_x, FOCAL_LENGTH_PX), math.atan2(error_y, FOCAL_LENGTH_PX)


def estimate_distance(bbox_h_px):
    """YOLO bbox height (px) -> target distance (mm). Falls back to 600mm."""
    if bbox_h_px <= 0:
        return 600.0
    return (TARGET_HEIGHT_MM * FOCAL_LENGTH_PX) / bbox_h_px


def angles_to_motor_delta(alpha_x, alpha_y, d_mm, current_tilt_rad):
    """
    Camera-frame angular offset -> motor angle deltas, corrected for the
    non-coaxial mount geometry.

    At close range both offsets cause the camera to translate as well as
    rotate, amplifying the effective pointing motion. We therefore command
    less motor movement than the raw angle suggests:
        k = 1 / (1 + offset/distance)
    """
    r_pan  = PAN_OFFSET_MM + TILT_OFFSET_MM * math.cos(current_tilt_rad)
    r_tilt = TILT_OFFSET_MM
    k_pan  = 1.0 / (1.0 + r_pan  / d_mm)
    k_tilt = 1.0 / (1.0 + r_tilt / d_mm)
    return alpha_x * k_pan, -alpha_y * k_tilt


# ── Serial helpers ────────────────────────────────────────────────────────────

def _compute_checksum(data):
    return sum(ord(c) for c in data) & 0xFF

def _build_packet(tilt, pan):
    data = f"T{tilt:.4f}P{pan:.4f}"
    return f"{data}:{_compute_checksum(data)}\n"

def _find_port():
    ports = list(serial.tools.list_ports.comports())
    esp = [p for p in ports if any(k in p.description for k in ("CP210", "CH340", "UART", "USB"))]
    if esp:
        return esp[0].device
    if ports:
        return ports[0].device
    raise RuntimeError("No serial port found - is the motor driver connected?")


class _SerialReader(threading.Thread):
    def __init__(self, ser):
        super().__init__(daemon=True)
        self.ser   = ser
        self.lines = []
        self._stop = False

    def run(self):
        while not self._stop:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode(errors="replace").strip()
                    if line:
                        self.lines.append(line)
                        if len(self.lines) > 30:
                            self.lines.pop(0)
            except Exception:
                break
            time.sleep(0.005)

    def stop(self):
        self._stop = True

    def wait_for(self, text, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(text in l for l in self.lines):
                return True
            time.sleep(0.02)
        return False


# ── Gimbal controller ─────────────────────────────────────────────────────────

class GimbalController:
    def __init__(self, port=None):
        port = port or _find_port()
        print(f"[Gimbal] Connecting to {port} @ {BAUD_RATE}...")
        self._ser    = serial.Serial(port, BAUD_RATE, timeout=0.1)
        self._reader = _SerialReader(self._ser)
        self._reader.start()

        self._ser.write(b"PING\n")
        if not self._reader.wait_for("READY", timeout=3.0):
            print("[Gimbal] WARN: no READY response - proceeding anyway")
        self._ser.write(b"ACK\n")
        self._ser.write(b"ENABLE\n")
        print("[Gimbal] Enabled.")

        self.actual_pan_rad  = 0.0
        self.actual_tilt_rad = 0.0

    def poll_actual(self):
        """Parse the most recent AT...P... feedback line and update actual angles."""
        for line in reversed(self._reader.lines):
            m = _ACTUAL_RE.search(line)
            if m:
                self.actual_tilt_rad = float(m.group(1))
                self.actual_pan_rad  = float(m.group(2))
                self._reader.lines.clear()
                return True
        return False

    def set_target(self, pan_rad, tilt_rad):
        """Send absolute target angles to the motor driver."""
        self._ser.write(_build_packet(tilt_rad, pan_rad).encode())

    def _send_zero(self):
        self._ser.write(_build_packet(0.0, 0.0).encode())

    def stop(self):
        self._ser.write(b"DISABLE\n")
        self._reader.stop()
        self._reader.join(timeout=1.0)
        self._ser.close()
        print("[Gimbal] Stopped.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        controls={"FrameRate": 120},
        transform=Transform(hflip=True, vflip=True)
    ))
    picam2.start()
    picam2.set_controls({"AfMode": 2})

    tracker = TrackerDisplay()
    gimbal  = GimbalController()

    prev_state      = TrackState.SEARCHING
    locked_dist_mm  = 600.0
    gimbal_active   = True
    last_gimbal_t   = time.time()
    integral_x      = 0.0
    integral_y      = 0.0
    smoothed_x      = 0.0
    smoothed_y      = 0.0
    prev_smoothed_x = 0.0
    prev_smoothed_y = 0.0

    try:
        while not tracker.should_stop():
            frame = picam2.capture_array()
            error_x, error_y, state = tracker.update(frame)

            key = tracker.last_key
            if key == ord("g") and not gimbal_active:
                gimbal_active = True
                print("[Gimbal] Tracking started.")
            elif key == ord("s") and gimbal_active:
                gimbal_active = False
                gimbal._send_zero()
                print("[Gimbal] Tracking stopped.")

            if prev_state != TrackState.TRACKING and state == TrackState.TRACKING:
                locked_dist_mm = estimate_distance(tracker.lock_bbox_h_px)
                smoothed_x, smoothed_y = pixel_to_angles(error_x, error_y)
                prev_smoothed_x, prev_smoothed_y = smoothed_x, smoothed_y
                integral_x, integral_y = 0.0, 0.0
                print(f"[Gimbal] Distance locked: {locked_dist_mm:.0f} mm")
            prev_state = state

            now = time.time()
            dt  = now - last_gimbal_t
            if gimbal_active and state == TrackState.TRACKING and dt >= SEND_INTERVAL:
                gimbal.poll_actual()

                alpha_x, alpha_y = pixel_to_angles(error_x, error_y)

                # EMA filter — smooths noise before D term differentiates it
                smoothed_x = EMA_ALPHA * alpha_x + (1 - EMA_ALPHA) * smoothed_x
                smoothed_y = EMA_ALPHA * alpha_y + (1 - EMA_ALPHA) * smoothed_y

                d_x = (smoothed_x - prev_smoothed_x) / dt
                d_y = (smoothed_y - prev_smoothed_y) / dt

                prev_smoothed_x = smoothed_x
                prev_smoothed_y = smoothed_y

                # integral — only accumulate outside dead zone, clamp to prevent windup
                if abs(error_x) > DEAD_ZONE_PX or abs(error_y) > DEAD_ZONE_PX:
                    integral_x += smoothed_x * dt
                    integral_y += smoothed_y * dt
                    integral_x = max(-INTEGRAL_CLAMP, min(INTEGRAL_CLAMP, integral_x))
                    integral_y = max(-INTEGRAL_CLAMP, min(INTEGRAL_CLAMP, integral_y))

                    pid_x = K_P_PAN  * smoothed_x + K_I_PAN  * integral_x + K_D_PAN  * d_x
                    pid_y = K_P_TILT * smoothed_y + K_I_TILT * integral_y + K_D_TILT * d_y
                    delta_pan, delta_tilt = angles_to_motor_delta(
                        pid_x, pid_y, locked_dist_mm, gimbal.actual_tilt_rad)
                    gimbal.set_target(
                        gimbal.actual_pan_rad  + delta_pan,
                        gimbal.actual_tilt_rad + delta_tilt)

                last_gimbal_t = now

    finally:
        gimbal.stop()
        tracker.stop()
        picam2.stop()


if __name__ == "__main__":
    main()
