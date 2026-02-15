"""
measure_axis.py — Measure total travel length of one axis (X or Y).

Run for each axis independently:
  1. You manually move the carriage to the FAR end (opposite the switch).
  2. You press Enter.
  3. The script runs the same homing move toward the switch.
  4. It records how far it traveled (mm) = total axis length.
  5. Saves to axis_lengths.json so the rest of the system knows your board size.

Usage:
    python3 measure_axis.py --port /dev/cu.usbmodem14101 --axis X
    python3 measure_axis.py --port /dev/cu.usbmodem14101 --axis Y
"""

import argparse
import json
import os
import time

import serial
from serial.tools import list_ports

# Same movement params as home.py
SEEK_FEED = 3500
SEEK_STEP = 4.0
BACKOFF_MM = 3.0
FINE_FEED = 800
FINE_STEP = 0.5
PULLOFF_MM = 1.5
MAX_TRAVEL = 450.0
MIRROR_Y_TO_Z = True

# Y-axis only: one smooth continuous move, higher power
Y_CONTINUOUS_FEED = 3500   # mm/min — smooth continuous move
Y_Z_SCALE = 0.90           # Z moves 90% of Y so Y motor (heavier side) leads

AXIS_LENGTHS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axis_lengths.json")


def send(ser, cmd, timeout=5.0):
    ser.reset_input_buffer()
    ser.write((cmd.strip() + "\n").encode())
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                low = line.lower()
                if low == "ok" or low.startswith("error") or low.startswith("alarm"):
                    break
                if line.startswith("<"):
                    break
        else:
            time.sleep(0.01)
    return ""


def get_status(ser):
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
    deadline = time.time() + timeout
    while time.time() < deadline:
        state, _, _, _ = get_status(ser)
        if state == "Idle":
            return True
        if "Alarm" in state:
            send(ser, "$X")
        time.sleep(0.05)
    return False


def jog_step(ser, dx_mm, dy_mm, feed):
    z_part = f" Z{dy_mm:.3f}" if MIRROR_Y_TO_Z and abs(dy_mm) > 1e-6 else ""
    send(ser, f"G1 X{dx_mm:.3f} Y{dy_mm:.3f}{z_part} F{feed:.0f}")
    wait_idle(ser, timeout=5)


def seek_until_pin(ser, pin, dx, dy, step_mm, feed, max_mm, label):
    """
    Jog toward switch until `pin` appears in Pn:.
    Returns (found: bool, traveled_mm: float).
    """
    traveled = 0.0
    send(ser, "G91")
    while traveled < max_mm:
        state, pins, x, y = get_status(ser)
        if pin in pins:
            send(ser, "G90")
            print(f"    {label} switch HIT at {traveled:.1f}mm")
            return True, traveled
        if "Alarm" in state:
            send(ser, "$X")
            send(ser, "G91")
        jog_step(ser, dx * step_mm, dy * step_mm, feed)
        traveled += step_mm
        if int(traveled) % 50 == 0 and int(traveled) > 0:
            print(f"    ... {traveled:.0f}mm", flush=True)
    send(ser, "G90")
    return False, traveled


def seek_continuous(ser, pin, dx, dy, label, feed, max_mm):
    """
    One smooth continuous move toward a switch until `pin` appears or alarm fires.
    Works for any axis (X or Y).  For Y, mirrors to Z with Y_Z_SCALE.
    Returns (found: bool, traveled_mm: float).
    """
    # Read starting position
    _, _, start_x, start_y = get_status(ser)
    axis_label = "X" if dx != 0 else "Y"
    start_val = start_x if dx != 0 else start_y
    print(f"    start pos: {axis_label}={start_val:.1f}")

    # Build the G1 command in relative mode
    send(ser, "G91")
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
    cmd = f"G1 {' '.join(parts)} F{feed:.0f}"
    send(ser, cmd)  # waits for "ok" — command is in GRBL's buffer

    # Give GRBL a moment to start executing (Idle → Run)
    time.sleep(0.3)

    # Poll for switch hit.  Must see "Run" before trusting "Idle".
    hit = False
    saw_run = False
    idle_count = 0

    while True:
        state, pins, x, y = get_status(ser)
        cur_val = x if dx != 0 else y

        if state in ("Run", "Jog"):
            saw_run = True
            idle_count = 0

        # Switch hit via pin
        if pin in pins:
            hit = True
            ser.write(b"!")
            print(f"    {label} switch HIT (pin) at {axis_label}={cur_val:.1f}")
            break

        # Switch hit via alarm
        if "Alarm" in state:
            hit = True
            print(f"    {label} switch HIT (alarm) at {axis_label}={cur_val:.1f}")
            break

        # Only trust Idle after move has started
        if state == "Idle" and saw_run:
            idle_count += 1
            if idle_count >= 3:
                print(f"    Move completed (Idle) at {axis_label}={cur_val:.1f}")
                break

        time.sleep(0.03)

    # Wait for machine to fully stop
    for _ in range(100):
        state, _, _, _ = get_status(ser)
        if state in ("Idle", "Hold", "Hold:0", "Hold:1"):
            break
        if "Alarm" in state:
            break
        time.sleep(0.05)

    # Read final position
    _, _, end_x, end_y = get_status(ser)
    end_val = end_x if dx != 0 else end_y
    print(f"    end pos: {axis_label}={end_val:.1f}")

    # Soft-reset to clear any remaining move in buffer
    ser.write(b"\x18")
    time.sleep(1.0)
    while ser.in_waiting:
        ser.readline()
    send(ser, "$X")
    send(ser, "G90")

    traveled = abs(end_val - start_val)
    return hit, traveled


def measure_axis(ser, axis):
    """
    Run homing toward the switch for one axis; return total travel in mm.
    Both axes use one smooth continuous move at SEEK_FEED.
    """
    if axis.upper() == "X":
        pin, dx, dy, label = "X", 1, 0, "X-left"
    else:
        pin, dx, dy, label = "Y", 0, 1, "Y-front"

    # Phase 1: one continuous move toward switch
    print(f"  Phase 1: continuous move toward {label} (F{SEEK_FEED})...")
    found, phase1_mm = seek_continuous(ser, pin, dx, dy, label, SEEK_FEED, MAX_TRAVEL)
    if not found:
        # Try opposite direction
        print(f"  Trying opposite direction...")
        found, phase1_mm = seek_continuous(ser, pin, -dx, -dy, label, SEEK_FEED, MAX_TRAVEL)
        if found:
            dx, dy = -dx, -dy
    if not found:
        print("  ERROR: Switch not found.")
        return None

    total_travel_mm = phase1_mm

    # Back off
    print(f"  Backing off {BACKOFF_MM}mm...")
    send(ser, "G91")
    z_part = ""
    if MIRROR_Y_TO_Z and abs(dy) > 1e-6:
        z_part = f" Z{-dy * BACKOFF_MM * Y_Z_SCALE:.3f}"
    send(ser, f"G1 X{-dx * BACKOFF_MM:.3f} Y{-dy * BACKOFF_MM:.3f}{z_part} F{SEEK_FEED:.0f}")
    wait_idle(ser, timeout=5)
    send(ser, "G90")

    # Phase 2: slow approach (stepped — short distance, precision matters)
    print(f"  Phase 2: slow approach (F{FINE_FEED})...")
    found2, _ = seek_until_pin(ser, pin, dx, dy, FINE_STEP, FINE_FEED, BACKOFF_MM + 5.0, label)
    if not found2:
        print("  WARNING: Switch not hit on slow approach.")

    # Pull-off
    print(f"  Pull-off {PULLOFF_MM}mm...")
    send(ser, "G91")
    z_part = ""
    if MIRROR_Y_TO_Z and abs(dy) > 1e-6:
        z_part = f" Z{-dy * PULLOFF_MM * Y_Z_SCALE:.3f}"
    send(ser, f"G1 X{-dx * PULLOFF_MM:.3f} Y{-dy * PULLOFF_MM:.3f}{z_part} F{FINE_FEED:.0f}")
    wait_idle(ser, timeout=5)
    send(ser, "G90")

    # Set origin
    if axis.upper() == "X":
        send(ser, "G92 X0")
    else:
        send(ser, "G92 Y0 Z0" if MIRROR_Y_TO_Z else "G92 Y0")

    return total_travel_mm


def load_lengths():
    if os.path.isfile(AXIS_LENGTHS_FILE):
        with open(AXIS_LENGTHS_FILE) as f:
            return json.load(f)
    return {}


def save_lengths(data):
    with open(AXIS_LENGTHS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {AXIS_LENGTHS_FILE}")


def find_port():
    for p in list_ports.comports():
        if any(k in p.description.lower() for k in ["arduino", "usbmodem", "usbserial", "ch340", "ftdi"]):
            return p.device
    return None


def main():
    ap = argparse.ArgumentParser(description="Measure axis travel length (far end → switch)")
    ap.add_argument("--port", type=str, default=None)
    ap.add_argument("--axis", choices=["x", "y", "X", "Y"], required=True)
    args = ap.parse_args()

    axis = args.axis.upper()
    port = args.port or find_port()
    if not port:
        print("ERROR: No Arduino found. Use --port")
        return

    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(2)
    while ser.in_waiting:
        ser.readline()

    print(f"Connected to {port}\n")

    try:
        send(ser, "$X")
        send(ser, "$5=1")
        send(ser, "$21=0")
        # Set max rates and acceleration high enough for our feed rates
        send(ser, "$110=6000")   # X max rate mm/min
        send(ser, "$111=6000")   # Y max rate mm/min
        send(ser, "$112=6000")   # Z max rate mm/min
        send(ser, "$120=200")    # X accel mm/s^2
        send(ser, "$121=200")    # Y accel mm/s^2
        send(ser, "$122=200")    # Z accel mm/s^2
        send(ser, "$X")
        send(ser, "G90")
        print("  GRBL max rates set to 6000mm/min, accel 200mm/s²")

        print("=" * 55)
        if axis == "X":
            print("  MEASURE X AXIS (left/right)")
            print("  Move the X carriage to the RIGHT (far end, opposite the left switch).")
        else:
            print("  MEASURE Y AXIS (front/back)")
            print("  Move the Y carriage to the BACK (far end, opposite the front switch).")
        print("=" * 55)
        input("  When ready, press Enter to start homing toward the switch... ")
        print()

        length_mm = measure_axis(ser, axis)
        if length_mm is None:
            return

        print(f"\n  Total travel (axis length): {length_mm:.1f} mm")

        # Merge into existing file
        data = load_lengths()
        data["x_mm" if axis == "X" else "y_mm"] = round(length_mm, 1)
        save_lengths(data)

        print(f"\n  Current axis_lengths.json: {data}")
        send(ser, "$21=1")
        send(ser, "$X")

    except KeyboardInterrupt:
        print("\nAborted.")
        ser.write(b"!")
    finally:
        ser.close()
        print("Disconnected.")


if __name__ == "__main__":
    main()
