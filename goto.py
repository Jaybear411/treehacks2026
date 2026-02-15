"""
goto.py — Move gantry to a point specified in a 400×400 virtual coordinate space.

Maps (0-400, 0-400) inputs to real mm using measured axis lengths from
axis_lengths.json. Tracks cumulative position from origin (0,0) and
clamps so the gantry never exceeds its physical travel in any direction.

The origin (0,0) in the 400-space maps to (0,0) mm — the homed front-left corner.
(400,400) maps to (x_mm, y_mm) — the full extent of each axis.

Usage:
    python3 goto.py --port /dev/cu.usbmodem14101 --x 200 --y 200
    python3 goto.py --x 0 --y 0                  # return to origin
    python3 goto.py --interactive                 # keep entering points

Can also be imported:
    from goto import GantryMover
    mover = GantryMover(port="/dev/cu.usbmodem14101")
    mover.connect()
    mover.goto(200, 200)   # center of 400x400 space
    mover.goto(0, 0)       # back to origin
    mover.disconnect()
"""

import argparse
import json
import os
import time

import serial
from serial.tools import list_ports

# ── Config ─────────────────────────────────────────────────────────────
VIRTUAL_SIZE = 400.0      # virtual coordinate space is 400 x 400
FEED_RATE = 3500          # mm/min for moves
MIRROR_Y_TO_Z = True
Y_Z_SCALE = 0.90          # Z gets 90% of Y so Y motor leads

AXIS_LENGTHS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axis_lengths.json")


def load_axis_lengths():
    if not os.path.isfile(AXIS_LENGTHS_FILE):
        raise FileNotFoundError(
            f"No axis_lengths.json found. Run measure_axis.py first.")
    with open(AXIS_LENGTHS_FILE) as f:
        data = json.load(f)
    x_mm = data.get("x_mm")
    y_mm = data.get("y_mm")
    if x_mm is None or y_mm is None:
        raise ValueError(
            f"axis_lengths.json missing x_mm or y_mm. Run measure_axis.py for both axes.")
    return float(x_mm), float(y_mm)


# ── Serial helpers ─────────────────────────────────────────────────────

def _find_port():
    for p in list_ports.comports():
        if any(k in p.description.lower()
               for k in ["arduino", "usbmodem", "usbserial", "ch340", "ftdi"]):
            return p.device
    return None


class GantryMover:
    """Move gantry in a 400×400 virtual space, tracking real mm position."""

    def __init__(self, port=None, feed_rate=FEED_RATE):
        self.port = port or _find_port()
        self.feed_rate = feed_rate
        self.ser = None

        # Measured physical limits (mm)
        self.x_max_mm, self.y_max_mm = load_axis_lengths()

        # Scale factors: virtual units → mm
        self.x_scale = self.x_max_mm / VIRTUAL_SIZE
        self.y_scale = self.y_max_mm / VIRTUAL_SIZE

        # Current position in real mm (starts at origin after homing)
        self.pos_x_mm = 0.0
        self.pos_y_mm = 0.0

        print(f"  Axis lengths: X={self.x_max_mm:.1f}mm, Y={self.y_max_mm:.1f}mm")
        print(f"  Scale: 1 virtual unit = {self.x_scale:.3f}mm (X), {self.y_scale:.3f}mm (Y)")

    # ── Serial ─────────────────────────────────────────────────────

    def connect(self):
        if not self.port:
            raise RuntimeError("No Arduino found. Specify port.")
        self.ser = serial.Serial(self.port, 115200, timeout=1)
        time.sleep(2)
        while self.ser.in_waiting:
            self.ser.readline()
        self._send("$X")
        self._send("$5=1")
        self._send("$21=0")       # hard limits off during moves
        self._send("$110=6000")
        self._send("$111=6000")
        self._send("$112=6000")
        self._send("$120=200")
        self._send("$121=200")
        self._send("$122=200")
        self._send("$X")
        self._send("G90")
        self._send("G92 X0 Y0 Z0")  # current pos = origin
        self.pos_x_mm = 0.0
        self.pos_y_mm = 0.0
        print(f"  Connected to {self.port} — origin set at current position")

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("  Disconnected.")

    def _send(self, cmd, timeout=5.0):
        self.ser.reset_input_buffer()
        self.ser.write((cmd.strip() + "\n").encode())
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ser.in_waiting:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    low = line.lower()
                    if low == "ok" or low.startswith("error") or low.startswith("alarm"):
                        return line
                    if line.startswith("<"):
                        return line
            else:
                time.sleep(0.01)
        return ""

    def _wait_idle(self, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.ser.write(b"?")
            time.sleep(0.05)
            while self.ser.in_waiting:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("<") and "Idle" in line:
                    return True
                if "Alarm" in line:
                    self._send("$X")
            time.sleep(0.05)
        return False

    # ── Core movement ──────────────────────────────────────────────

    def goto(self, vx: float, vy: float):
        """
        Move to (vx, vy) in the 400×400 virtual space.

        Positive values are absolute positions (0-400).
        Negative values are relative: move back toward origin by that amount.
          e.g. goto(-200, 0) from pos (300, 100) → go to (100, 100).
        Movement toward origin is clamped so you can't go below 0 (origin).
        Movement away from origin is clamped at axis max.
        """
        # Handle negative as relative (subtract from current virtual position)
        cur_vx = self.pos_x_mm / self.x_scale
        cur_vy = self.pos_y_mm / self.y_scale

        if vx < 0:
            vx = cur_vx + vx  # e.g. 300 + (-200) = 100
        if vy < 0:
            vy = cur_vy + vy

        # Convert virtual → mm
        target_x_mm = vx * self.x_scale
        target_y_mm = vy * self.y_scale

        # Clamp to physical limits [0, max]
        target_x_mm = max(0.0, min(self.x_max_mm, target_x_mm))
        target_y_mm = max(0.0, min(self.y_max_mm, target_y_mm))

        # Delta from current tracked position
        dx = target_x_mm - self.pos_x_mm
        dy = target_y_mm - self.pos_y_mm

        if abs(dx) < 0.01 and abs(dy) < 0.01:
            print(f"  Already at ({vx:.0f}, {vy:.0f})")
            return True

        print(f"  goto({vx:.0f}, {vy:.0f}) → "
              f"({target_x_mm:.1f}, {target_y_mm:.1f})mm  "
              f"Δ=({dx:+.1f}, {dy:+.1f})mm")

        # GRBL coordinates: origin (switches) is at (0,0) and the travel
        # area extends in the NEGATIVE direction (homing seeks +X/+Y toward
        # switches, so away from switches = negative).  Negate the targets.
        grbl_x = -target_x_mm
        grbl_y = -target_y_mm
        z_part = ""
        if MIRROR_Y_TO_Z:
            grbl_z = grbl_y * Y_Z_SCALE
            z_part = f" Z{grbl_z:.3f}"

        cmd = f"G1 X{grbl_x:.3f} Y{grbl_y:.3f}{z_part} F{self.feed_rate:.0f}"
        resp = self._send(cmd)

        if "error" in resp.lower() or "alarm" in resp.lower():
            print(f"  MOVE FAILED: {resp}")
            self._send("$X")
            return False

        self._wait_idle()

        # Update tracked position
        self.pos_x_mm = target_x_mm
        self.pos_y_mm = target_y_mm

        print(f"  Position: ({self.pos_x_mm:.1f}, {self.pos_y_mm:.1f})mm  "
              f"[{self.pos_x_mm / self.x_scale:.0f}, {self.pos_y_mm / self.y_scale:.0f} virtual]")
        return True

    def go_home(self):
        """Return to origin (0, 0)."""
        return self.goto(0, 0)

    @property
    def virtual_position(self):
        """Current position in the 400×400 virtual space."""
        return (self.pos_x_mm / self.x_scale, self.pos_y_mm / self.y_scale)


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Move gantry to a point in 400×400 virtual space")
    ap.add_argument("--port", type=str, default=None)
    ap.add_argument("--x", type=float, default=None, help="X coord (0-400)")
    ap.add_argument("--y", type=float, default=None, help="Y coord (0-400)")
    ap.add_argument("--feed", type=float, default=FEED_RATE, help="Feed rate mm/min")
    ap.add_argument("--interactive", action="store_true",
                    help="Keep entering coordinates")
    args = ap.parse_args()

    mover = GantryMover(port=args.port, feed_rate=args.feed)
    mover.connect()

    try:
        if args.interactive or (args.x is None and args.y is None):
            print("\n" + "=" * 55)
            print(f"  INTERACTIVE MODE — enter coords in 0-400 space")
            print(f"  Physical: X=0..{mover.x_max_mm:.0f}mm, Y=0..{mover.y_max_mm:.0f}mm")
            print(f"  Type 'h' for home, 'q' to quit")
            print("=" * 55 + "\n")

            while True:
                try:
                    raw = input("  x y > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break
                if raw in ("q", "quit", "exit"):
                    break
                if raw in ("h", "home"):
                    mover.go_home()
                    continue
                parts = raw.replace(",", " ").split()
                if len(parts) != 2:
                    print("  Enter two numbers: x y  (0-400)")
                    continue
                try:
                    vx, vy = float(parts[0]), float(parts[1])
                except ValueError:
                    print("  Invalid numbers.")
                    continue
                mover.goto(vx, vy)
        else:
            vx = args.x if args.x is not None else 0
            vy = args.y if args.y is not None else 0
            mover.goto(vx, vy)

    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        mover.disconnect()


if __name__ == "__main__":
    main()
