"""
home.py — Move gantry to the front-left origin using limit switches.

Smooth continuous moves (not stepped) — same feel as goto.py.

Two-phase homing per axis:
  1. One smooth continuous move toward switch (fast)
  2. Back off
  3. Slow approach for precision
  4. Pull-off
  5. Set (0, 0)

Switches (CNC Shield V3):
  Y-axis: front endstop → GRBL Y pin
  X-axis: left endstop  → GRBL X pin

Dual-Y: mirrors Y moves to Z axis automatically.

Usage:
    python3 home.py --port /dev/cu.usbmodem14101
    python3 home.py                               # auto-detect port
"""

import argparse
import time

import serial
from serial.tools import list_ports

# ── Homing config ──────────────────────────────────────────────────────
SEEK_FEED = 3500      # mm/min — fast continuous seek
FINE_FEED = 800       # mm/min — slow precision approach
FINE_STEP = 0.5       # mm per jog during slow approach only
BACKOFF_MM = 3.0      # mm to back off after fast seek hit
PULLOFF_MM = 1.5      # mm final pull-off from switch (where 0,0 is set)
MAX_TRAVEL = 450.0    # mm max before giving up
MIRROR_Y_TO_Z = True  # dual-Y: copy Y moves to Z axis
Y_Z_SCALE = 0.90      # Z gets 90% of Y so Y motor leads


# ── Serial helpers ─────────────────────────────────────────────────────

def send(ser, cmd, timeout=5.0):
    """Send a GRBL command, return response."""
    ser.reset_input_buffer()
    ser.write((cmd.strip() + "\n").encode())
    lines = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                lines.append(line)
                low = line.lower()
                if low == "ok" or low.startswith("error") or low.startswith("alarm"):
                    break
                if line.startswith("<"):
                    break
        else:
            time.sleep(0.01)
    return "\n".join(lines)


def get_status(ser):
    """Query GRBL status. Returns (state, pins, x, y)."""
    ser.write(b"?")
    deadline = time.time() + 0.5
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("<") and ">" in line:
                inner = line[1:line.index(">")]
                fields = inner.split("|")
                state = fields[0]
                pins, x, y = "", 0.0, 0.0
                for f in fields[1:]:
                    if f.startswith("MPos:") or f.startswith("WPos:"):
                        coords = f.split(":")[1].split(",")
                        x, y = float(coords[0]), float(coords[1])
                    elif f.startswith("Pn:"):
                        pins = f[3:]
                return state, pins, x, y
        else:
            time.sleep(0.01)
    return "?", "", 0.0, 0.0


def wait_idle(ser, timeout=10.0):
    """Wait until GRBL reports Idle."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        state, _, _, _ = get_status(ser)
        if state == "Idle":
            return True
        if "Alarm" in state:
            send(ser, "$X")
        time.sleep(0.05)
    return False


# ── Continuous seek ────────────────────────────────────────────────────

def seek_continuous(ser, pin, dx, dy, feed, max_mm, label):
    """
    One smooth continuous move until `pin` appears or alarm fires.
    Returns True if switch was hit.
    """
    # Build G1 command
    x_dist = dx * max_mm
    y_dist = dy * max_mm
    z_dist = y_dist * Y_Z_SCALE if MIRROR_Y_TO_Z and abs(y_dist) > 1e-6 else 0
    parts = []
    if abs(x_dist) > 1e-6:
        parts.append(f"X{x_dist:.1f}")
    if abs(y_dist) > 1e-6:
        parts.append(f"Y{y_dist:.1f}")
    if abs(z_dist) > 1e-6:
        parts.append(f"Z{z_dist:.1f}")

    send(ser, "G91")
    cmd = f"G1 {' '.join(parts)} F{feed:.0f}"
    send(ser, cmd)

    # Give GRBL a moment to start (Idle → Run)
    time.sleep(0.3)

    hit = False
    saw_run = False
    idle_count = 0

    while True:
        state, pins, x, y = get_status(ser)

        if state in ("Run", "Jog"):
            saw_run = True
            idle_count = 0

        # Switch hit via pin
        if pin in pins:
            hit = True
            ser.write(b"!")
            print(f"    {label} switch HIT (pin)")
            break

        # Switch hit via alarm
        if "Alarm" in state:
            hit = True
            print(f"    {label} switch HIT (alarm)")
            break

        # Only trust Idle after move started
        if state == "Idle" and saw_run:
            idle_count += 1
            if idle_count >= 3:
                break

        time.sleep(0.03)

    # Wait for full stop
    for _ in range(100):
        state, _, _, _ = get_status(ser)
        if state in ("Idle", "Hold", "Hold:0", "Hold:1"):
            break
        if "Alarm" in state:
            break
        time.sleep(0.05)

    # Soft-reset to clear remaining move
    ser.write(b"\x18")
    time.sleep(1.0)
    while ser.in_waiting:
        ser.readline()
    send(ser, "$X")
    send(ser, "G90")

    return hit


def seek_stepped(ser, pin, dx, dy, step_mm, feed, max_mm, label):
    """Stepped seek for short-distance precision approach."""
    traveled = 0.0
    send(ser, "G91")
    while traveled < max_mm:
        state, pins, x, y = get_status(ser)
        if pin in pins:
            send(ser, "G90")
            return True
        if "Alarm" in state:
            send(ser, "$X")
            send(ser, "G91")
        z_part = ""
        if MIRROR_Y_TO_Z and abs(dy) > 1e-6:
            z_part = f" Z{dy * step_mm * Y_Z_SCALE:.3f}"
        send(ser, f"G1 X{dx * step_mm:.3f} Y{dy * step_mm:.3f}{z_part} F{feed:.0f}")
        wait_idle(ser, timeout=5)
        traveled += step_mm
    send(ser, "G90")
    return False


# ── Home one axis ──────────────────────────────────────────────────────

def home_axis(ser, pin, axis_name, dx, dy):
    """
    Two-phase homing:
      Phase 1: smooth continuous seek toward switch
      Phase 2: back off, slow stepped approach, pull off
    """
    # Phase 1: fast continuous seek
    print(f"\n  [{axis_name}] Seeking switch (F{SEEK_FEED})...")
    found = seek_continuous(ser, pin, dx, dy, SEEK_FEED, MAX_TRAVEL, axis_name)

    if not found:
        print(f"  [{axis_name}] Trying opposite direction...")
        found = seek_continuous(ser, pin, -dx, -dy, SEEK_FEED, MAX_TRAVEL, axis_name)
        if found:
            dx, dy = -dx, -dy

    if not found:
        print(f"  [{axis_name}] WARNING: switch not found!")
        return False

    # Back off
    print(f"  [{axis_name}] Backing off {BACKOFF_MM}mm...")
    send(ser, "G91")
    z_part = ""
    if MIRROR_Y_TO_Z and abs(dy) > 1e-6:
        z_part = f" Z{-dy * BACKOFF_MM * Y_Z_SCALE:.3f}"
    send(ser, f"G1 X{-dx * BACKOFF_MM:.3f} Y{-dy * BACKOFF_MM:.3f}{z_part} F{SEEK_FEED:.0f}")
    wait_idle(ser, timeout=5)
    send(ser, "G90")

    # Phase 2: slow stepped approach (short distance, precision)
    print(f"  [{axis_name}] Slow approach (F{FINE_FEED})...")
    found = seek_stepped(ser, pin, dx, dy, FINE_STEP, FINE_FEED,
                         BACKOFF_MM + 5.0, axis_name)
    if not found:
        print(f"  [{axis_name}] WARNING: switch not hit on slow approach")
        return False

    # Pull-off
    print(f"  [{axis_name}] Pull-off {PULLOFF_MM}mm...")
    send(ser, "G91")
    z_part = ""
    if MIRROR_Y_TO_Z and abs(dy) > 1e-6:
        z_part = f" Z{-dy * PULLOFF_MM * Y_Z_SCALE:.3f}"
    send(ser, f"G1 X{-dx * PULLOFF_MM:.3f} Y{-dy * PULLOFF_MM:.3f}{z_part} F{FINE_FEED:.0f}")
    wait_idle(ser, timeout=5)
    send(ser, "G90")

    print(f"  [{axis_name}] Done.")
    return True


# ── Main ───────────────────────────────────────────────────────────────

def find_port():
    for p in list_ports.comports():
        if any(k in p.description.lower()
               for k in ['arduino', 'usbmodem', 'usbserial', 'ch340', 'ftdi']):
            return p.device
    return None


def main():
    global SEEK_FEED, FINE_FEED, MAX_TRAVEL

    ap = argparse.ArgumentParser(description="Home gantry to front-left origin")
    ap.add_argument("--port", type=str, default=None)
    ap.add_argument("--seek-feed", type=float, default=SEEK_FEED)
    ap.add_argument("--fine-feed", type=float, default=FINE_FEED)
    ap.add_argument("--max-travel", type=float, default=MAX_TRAVEL)
    args = ap.parse_args()

    SEEK_FEED = args.seek_feed
    FINE_FEED = args.fine_feed
    MAX_TRAVEL = args.max_travel

    port = args.port or find_port()
    if not port:
        print("ERROR: No Arduino found. Specify --port")
        return

    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(2)
    while ser.in_waiting:
        ser.readline()

    print(f"Connected to {port}")

    try:
        # ── Safe starting state ────────────────────────────────
        send(ser, "$X")
        send(ser, "$5=1")
        send(ser, "$21=0")
        send(ser, "$110=6000")
        send(ser, "$111=6000")
        send(ser, "$112=6000")
        send(ser, "$120=200")
        send(ser, "$121=200")
        send(ser, "$122=200")
        send(ser, "$X")
        send(ser, "G90")
        send(ser, "G92 X0 Y0 Z0")

        print("\n" + "=" * 55)
        print("  HOMING TO FRONT-LEFT ORIGIN")
        print("  Y → front endstop (GRBL Y pin)")
        print("  X → left endstop  (GRBL X pin)")
        print("=" * 55)

        # ── Home Y first (toward front = positive Y) ──────────
        y_ok = home_axis(ser, pin="Y", axis_name="Y-front",
                         dx=0, dy=1)

        # ── Home X (toward left = positive X) ─────────────────
        x_ok = home_axis(ser, pin="X", axis_name="X-left",
                         dx=1, dy=0)

        # ── Set origin ─────────────────────────────────────────
        send(ser, "G92 X0 Y0 Z0")

        send(ser, "$21=1")
        send(ser, "$X")

        state, pins, x, y = get_status(ser)
        print(f"\n{'=' * 55}")
        if x_ok and y_ok:
            print(f"  HOMED SUCCESSFULLY — Origin set to (0, 0)")
        else:
            print(f"  PARTIAL HOME — Origin set at current position")
        print(f"  Position: ({x:.1f}, {y:.1f})")
        print(f"  Hard limits: ON ($21=1)")
        print(f"{'=' * 55}")

    except KeyboardInterrupt:
        print("\n\nAborted — stopping motors...")
        ser.write(b"!")
        time.sleep(0.5)
        ser.write(b"\x18")
        time.sleep(1)
        send(ser, "$X")
        print("Motors stopped.")
    finally:
        ser.close()
        print("Disconnected.")


if __name__ == "__main__":
    main()
