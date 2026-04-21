"""
Motor Control State Machine Tester
-----------------------------------
Tests the full state sequence:
  BOOT → CALIBRATING → READY → [ACK] → TRACKING

Commands:
  CAL     → re-triggers calibration from TRACKING or DISABLED
  DISABLE → disables motors (from TRACKING)
  ENABLE  → re-enables motors and returns to TRACKING (from DISABLED)

All angle values are in radians, matching the MCU protocol.
Pan is an offset from boot position; tilt is an offset from calibrated center.

Usage:
  python test_motor_states.py               # auto-detect port
  python test_motor_states.py /dev/ttyUSB0  # specify port
"""

import serial
import serial.tools.list_ports
import time
import sys
import threading
import math

# ─── Config ───────────────────────────────────────────────────────────────────
BAUD_RATE       = 115200
PACKET_INTERVAL = 0.1    # seconds between tracking packets
CAL_TIMEOUT_S   = 30     # max time to wait for calibration

# ─── Helpers ──────────────────────────────────────────────────────────────────
def compute_checksum(data: str) -> int:
    return sum(ord(c) for c in data) & 0xFF

def build_packet(tilt: float, pan: float) -> str:
    data = f"T{tilt:.4f}P{pan:.4f}"
    return f"{data}:{compute_checksum(data)}\n"

def log(tag: str, msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}")

def deg(d: float) -> float:
    return math.radians(d)

def find_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    esp_ports = [p for p in ports if any(k in p.description for k in ("CP210", "CH340", "UART", "USB"))]
    if esp_ports:
        return esp_ports[0].device
    if ports:
        return ports[0].device
    raise RuntimeError("No serial ports found. Plug in your device or pass port as argument.")

# ─── Reader thread ────────────────────────────────────────────────────────────
class SerialReader(threading.Thread):
    def __init__(self, ser: serial.Serial):
        super().__init__(daemon=True)
        self.ser = ser
        self.lines = []
        self._stop = False

    def run(self):
        while not self._stop:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode(errors="replace").strip()
                    if line:
                        self.lines.append(line)
                        log("MCU", line)
            except Exception:
                break
            time.sleep(0.005)

    def stop(self):
        self._stop = True

    def wait_for(self, text: str, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(text in l for l in self.lines):
                return True
            time.sleep(0.05)
        return False

    def clear(self):
        self.lines.clear()

# ─── Test stages ──────────────────────────────────────────────────────────────

def stage_boot(reader: SerialReader):
    log("TEST", "── Stage: BOOT ──────────────────────────────────")
    log("TEST", "Waiting for firmware to start...")
    if reader.wait_for("LOG:", timeout=5.0):
        log("PASS", "Firmware is running — saw first LOG output.")
    else:
        log("WARN", "No LOG lines seen yet. Device may still be booting.")

def stage_calibrating(reader: SerialReader):
    log("TEST", "── Stage: CALIBRATING ───────────────────────────")
    log("TEST", f"Waiting up to {CAL_TIMEOUT_S}s for 'READY'...")
    reader.clear()

    start = time.time()
    if reader.wait_for("READY", timeout=CAL_TIMEOUT_S):
        elapsed = time.time() - start
        log("PASS", f"Calibration complete in {elapsed:.1f}s — received 'READY'.")
    else:
        log("FAIL", "Timed out waiting for 'READY'. Check wiring / endstops.")
        sys.exit(1)

def stage_ready(ser: serial.Serial, reader: SerialReader):
    log("TEST", "── Stage: READY ─────────────────────────────────")
    log("TEST", "Sending 'ACK' to transition to TRACKING...")
    reader.clear()
    ser.write(b"ACK\n")
    time.sleep(0.2)
    log("PASS", "ACK sent — now in TRACKING.")

def stage_tracking(ser: serial.Serial, reader: SerialReader):
    log("TEST", "── Stage: TRACKING ──────────────────────────────")
    log("INFO", "All angles in radians. Tilt = offset from cal center, Pan = offset from boot position.")

    test_moves = [
        (0.0,       0.0,      "center / home"),
        (deg(10),   0.0,      "tilt +10°"),
        (deg(-10),  0.0,      "tilt -10°"),
        (0.0,       deg(20),  "pan +20°"),
        (0.0,       deg(-20), "pan -20°"),
        (deg(15),   deg(30),  "combined +15° tilt / +30° pan"),
        (deg(-15),  deg(-30), "combined -15° tilt / -30° pan"),
        (0.0,       0.0,      "return to center"),
    ]

    for tilt, pan, label in test_moves:
        pkt = build_packet(tilt, pan)
        log("SEND", f"T={tilt:+.4f} rad  P={pan:+.4f} rad  ({label})")
        ser.write(pkt.encode())
        time.sleep(2.0)

    log("PASS", "Discrete tracking moves complete.")

def stage_continuous_sweep(ser: serial.Serial, reader: SerialReader, duration_s: float = 5.0):
    log("TEST", "── Stage: Continuous pan sweep ──────────────────")
    max_pan = deg(30)
    log("INFO", f"Sweeping pan ±{math.degrees(max_pan):.1f}° ({max_pan:.4f} rad) over {duration_s}s.")

    start = time.time()
    while time.time() - start < duration_s:
        t = time.time() - start
        pan = max_pan * math.sin(2 * math.pi * (t / duration_s))
        ser.write(build_packet(0.0, pan).encode())
        time.sleep(PACKET_INTERVAL)

    ser.write(build_packet(0.0, 0.0).encode())
    log("PASS", "Continuous sweep done, parked at center.")

def stage_disable_enable(ser: serial.Serial, reader: SerialReader):
    log("TEST", "── Stage: DISABLE / ENABLE ──────────────────────")

    log("TEST", "Sending DISABLE...")
    ser.write(b"DISABLE\n")
    time.sleep(2.0)
    log("PASS", "Motors should be disabled. Verify no movement on physical unit.")

    log("TEST", "Sending ENABLE — should return to TRACKING...")
    ser.write(b"ENABLE\n")
    time.sleep(0.5)

    # Send a packet immediately to confirm tracking resumed
    pkt = build_packet(0.0, 0.0)
    ser.write(pkt.encode())
    time.sleep(1.0)
    log("PASS", "ENABLE sent, tracking packet sent — motors should be live again.")

def stage_recalibrate(ser: serial.Serial, reader: SerialReader):
    log("TEST", "── Stage: CAL command (recalibrate) ─────────────")
    log("TEST", "Sending CAL from TRACKING — should restart calibration...")
    reader.clear()
    ser.write(b"CAL\n")

    start = time.time()
    if reader.wait_for("READY", timeout=CAL_TIMEOUT_S):
        elapsed = time.time() - start
        log("PASS", f"Recalibration complete in {elapsed:.1f}s.")
    else:
        log("FAIL", "Timed out waiting for 'READY' after CAL command.")
        sys.exit(1)

    # Re-ACK after recal
    log("TEST", "Sending ACK to re-enter TRACKING after recal...")
    ser.write(b"ACK\n")
    time.sleep(0.2)
    log("PASS", "Back in TRACKING.")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_port()
    log("INFO", f"Connecting to {port} @ {BAUD_RATE} baud...")

    with serial.Serial(port, BAUD_RATE, timeout=0.1) as ser:
        time.sleep(1.5)
        ser.reset_input_buffer()

        reader = SerialReader(ser)
        reader.start()

        try:
            stage_boot(reader)
            stage_calibrating(reader)
            stage_ready(ser, reader)
            stage_tracking(ser, reader)
            stage_continuous_sweep(ser, reader, duration_s=5.0)
            stage_disable_enable(ser, reader)
            stage_recalibrate(ser, reader)

            # Final sweep after recal to confirm everything still works
            log("TEST", "── Final tracking check after recal ─────────────")
            stage_continuous_sweep(ser, reader, duration_s=3.0)

            log("TEST", "════════════════════════════════════════════════")
            log("TEST", "All stages complete!")

        except KeyboardInterrupt:
            log("INFO", "Interrupted by user.")
            ser.write(build_packet(0.0, 0.0).encode())
        finally:
            reader.stop()
            reader.join(timeout=1.0)

if __name__ == "__main__":
    main()