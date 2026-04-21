"""
Motor Driver Interactive Console
----------------------------------
Manual control for the pan/tilt motor driver.

Usage:
  python motor_console.py               # auto-detect port
  python motor_console.py /dev/ttyUSB0  # specify port
"""

import serial
import serial.tools.list_ports
import time
import sys
import threading
import math

# ─── Config ───────────────────────────────────────────────────────────────────
BAUD_RATE = 115200

# ─── Helpers ──────────────────────────────────────────────────────────────────
def compute_checksum(data: str) -> int:
    return sum(ord(c) for c in data) & 0xFF

def build_packet(tilt: float, pan: float) -> str:
    data = f"T{tilt:.4f}P{pan:.4f}"
    return f"{data}:{compute_checksum(data)}\n"

def log(tag: str, msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}")

def find_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    esp_ports = [p for p in ports if any(k in p.description for k in ("CP210", "CH340", "UART", "USB"))]
    if esp_ports:
        return esp_ports[0].device
    if ports:
        return ports[0].device
    raise RuntimeError("No serial ports found.")

# ─── Serial reader ────────────────────────────────────────────────────────────
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

# ─── Commands ─────────────────────────────────────────────────────────────────
def cmd_ping(ser: serial.Serial, reader: SerialReader):
    log("SEND", "PING")
    reader.lines.clear()
    ser.write(b"PING\n")
    if reader.wait_for("READY", timeout=2.0):
        log("PASS", "Driver is READY.")
    else:
        log("WARN", "No READY response — driver may not be in READY state.")

def cmd_ack(ser: serial.Serial):
    log("SEND", "ACK")
    ser.write(b"ACK\n")

def cmd_move_tilt(ser: serial.Serial):
    try:
        deg = float(input("  Tilt angle (degrees): "))
        rad = math.radians(deg)
        pkt = build_packet(rad, 0.0)
        log("SEND", f"Tilt {deg:+.2f}° ({rad:+.4f} rad), Pan zeroed")
        ser.write(pkt.encode())
    except ValueError:
        log("ERR", "Invalid input — enter a number.")

def cmd_move_pan(ser: serial.Serial):
    try:
        deg = float(input("  Pan angle (degrees): "))
        rad = math.radians(deg)
        pkt = build_packet(0.0, rad)
        log("SEND", f"Pan {deg:+.2f}° ({rad:+.4f} rad), Tilt zeroed")
        ser.write(pkt.encode())
    except ValueError:
        log("ERR", "Invalid input — enter a number.")

def cmd_enable(ser: serial.Serial):
    log("SEND", "ENABLE")
    ser.write(b"ENABLE\n")

def cmd_disable(ser: serial.Serial):
    log("SEND", "DISABLE")
    ser.write(b"DISABLE\n")

def cmd_calibrate(ser: serial.Serial):
    log("SEND", "CAL")
    ser.write(b"CAL\n")

# ─── REPL ─────────────────────────────────────────────────────────────────────
MENU = """
  [1] ping        [2] ack
  [3] move tilt   [4] move pan
  [5] enable      [6] disable
  [7] calibrate   [q] quit
"""

def repl(ser: serial.Serial, reader: SerialReader):
    print(MENU)
    while True:
        try:
            choice = input(">> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if   choice == "1": cmd_ping(ser, reader)
        elif choice == "2": cmd_ack(ser)
        elif choice == "3": cmd_move_tilt(ser)
        elif choice == "4": cmd_move_pan(ser)
        elif choice == "5": cmd_enable(ser)
        elif choice == "6": cmd_disable(ser)
        elif choice == "7": cmd_calibrate(ser)
        elif choice == "q": break
        else: print(MENU)

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
            repl(ser, reader)
        finally:
            log("INFO", "Closing connection.")
            reader.stop()
            reader.join(timeout=1.0)

if __name__ == "__main__":
    main()