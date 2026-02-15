"""
Motor and limit switch test for GRBL 1.1.

Modes:
  1. Single-axis back-and-forth (default)
  2. --seek-limits: move X/Y diagonally until limit alarm OR max distance

Does NOT override $5 (limit pin invert) — configure that with switch_diag.py first.

Usage:
    python3 motortest.py --port /dev/cu.usbmodem14101 --axis x
    python3 motortest.py --port /dev/cu.usbmodem14101 --seek-limits --feed 1200
"""

import argparse
import time

from grbl_controller import GRBLController, GRBLState

# GRBL settings applied before each test (does NOT include $5 — that's a
# hardware config set once via switch_diag.py and saved to EEPROM).
GRBL_SETTINGS = [
    ("$0", "10",    "step pulse (us)"),
    ("$1", "25",    "step idle delay (ms)"),
    # $5 intentionally omitted — set once via switch_diag.py and saved to EEPROM.
    ("$21", "1",    "hard limits ON"),
    ("$100", "160", "X steps/mm"),
    ("$101", "160", "Y steps/mm"),
    ("$102", "160", "Z steps/mm"),
    ("$110", "2000", "X max rate (mm/min)"),
    ("$111", "2000", "Y max rate (mm/min)"),
    ("$112", "2000", "Z max rate (mm/min)"),
    ("$120", "80",  "X accel (mm/s^2)"),
    ("$121", "80",  "Y accel (mm/s^2)"),
    ("$122", "80",  "Z accel (mm/s^2)"),
]

# GRBL pin → human name
PIN_NAMES = {"X": "X", "Y": "Y+", "Z": "Y-"}


def apply_settings(ctrl):
    print("Applying GRBL settings...")
    for key, val, desc in GRBL_SETTINGS:
        cmd = f"{key}={val}"
        ctrl.serial.reset_input_buffer()
        ctrl.serial.write((cmd + "\n").encode())
        time.sleep(0.3)
        resp = ""
        while ctrl.serial.in_waiting:
            resp += ctrl.serial.readline().decode("utf-8", errors="ignore").strip() + " "
        ok = "ok" if "ok" in resp.lower() else resp.strip()
        print(f"  {cmd:12s} ({desc}) -> {ok}")


def wait_idle(ctrl, timeout=60.0):
    start = time.time()
    while (time.time() - start) < timeout:
        s = ctrl.get_status()
        if s:
            if s.state == GRBLState.IDLE:
                return True
            if s.state == GRBLState.ALARM:
                return False
        time.sleep(0.1)
    return False


def pins_to_names(pins):
    return ", ".join(PIN_NAMES.get(p, p) for p in pins) if pins else ""


def unlock(ctrl):
    print("  Unlocking ($X)...")
    ctrl.serial.reset_input_buffer()
    ctrl.serial.write(b"$X\n")
    time.sleep(0.5)
    while ctrl.serial.in_waiting:
        ctrl.serial.readline()
    ctrl.get_status()


# ── Diagonal limit-seek test ──────────────────────────────────────────

def check_limit(ctrl):
    """Return description if alarm or Y-limit pin active, else None."""
    s = ctrl.get_status()
    if not s:
        return None
    pins = s.active_pins
    if s.state == GRBLState.ALARM:
        name = pins_to_names(pins) or "ALARM"
        return f"{name} (ALARM, pos=({s.position_x:.1f}, {s.position_y:.1f}))"
    # Check for Y-related pins
    if "Y" in pins or "Z" in pins:
        name = pins_to_names(pins)
        return f"{name} (pos=({s.position_x:.1f}, {s.position_y:.1f}))"
    return None


def seek_limit(ctrl, sign, step, feed, max_dist):
    """
    Jog X+Y together until:
    - A Y limit alarm/pin fires, OR
    - We've traveled max_dist mm (software safety stop)
    """
    label = "Y+" if sign > 0 else "Y-"
    max_steps = int(max_dist / step) + 1
    print(f"Seeking {label}: step={sign * step:.1f}mm, feed=F{feed:.0f}, max={max_dist:.0f}mm")

    traveled = 0.0
    for idx in range(1, max_steps + 1):
        # Check before moving
        hit = check_limit(ctrl)
        if hit:
            print(f"  {label} limit at {traveled:.1f}mm: {hit}")
            return True

        ok = ctrl.jog(sign * step, sign * step, feed_rate=feed)
        traveled += step

        if not ok:
            hit = check_limit(ctrl)
            if hit:
                print(f"  {label} limit at {traveled:.1f}mm: {hit}")
                return True
            print(f"  Jog failed at {traveled:.1f}mm, unlocking...")
            unlock(ctrl)
            continue

        if idx % 20 == 0:
            s = ctrl.get_status()
            if s:
                active = pins_to_names(s.active_pins)
                pin_str = f"  pins={active}" if active else ""
                print(f"  {traveled:.0f}mm  pos=({s.position_x:.1f}, {s.position_y:.1f})"
                      f"  state={s.state.value}{pin_str}")

    print(f"  Reached max distance ({max_dist:.0f}mm) without hitting {label} limit.")
    print(f"  (This is OK if you don't have switches wired yet.)")
    return False


def run_seek_limits(ctrl, step, feed, max_dist):
    print("\n" + "=" * 55)
    print("  DIAGONAL XY LIMIT TEST")
    print("  Moves X+Y together, stops on limit alarm or max distance")
    print("=" * 55)

    ctrl.soft_home()
    ctrl._send_command("G90", wait_for_ok=True)

    # Seek Y+
    found_plus = seek_limit(ctrl, +1, step, feed, max_dist)
    if found_plus:
        print("  Y+ limit confirmed.\n")
        unlock(ctrl)
        # Back off
        print("Backing off from limit...")
        ctrl._send_command("G91", wait_for_ok=True)
        ctrl._send_command(f"G1 X{-step * 5:.3f} Y{-step * 5:.3f} F{feed:.0f}", wait_for_ok=True)
        ctrl._send_command("G90", wait_for_ok=True)
        wait_idle(ctrl, timeout=10)
    else:
        print("  Y+ limit not found (stopped at max distance).\n")

    ctrl.soft_home()

    # Seek Y-
    found_minus = seek_limit(ctrl, -1, step, feed, max_dist)
    if found_minus:
        print("  Y- limit confirmed.\n")
        unlock(ctrl)
    else:
        print("  Y- limit not found (stopped at max distance).\n")

    return found_plus or found_minus


# ── Single-axis back-and-forth ────────────────────────────────────────

def run_axis_test(ctrl, axes, distance, feed, loops):
    print(f"\nMotor test: axes={axes}  distance={distance}mm  feed={feed}mm/min  loops={loops}")
    print("Ctrl+C to stop.\n")

    try:
        for axis in axes:
            label = axis.upper()
            print(f"{'=' * 40}")
            print(f"  Testing {label} axis")
            print(f"{'=' * 40}")
            ctrl.soft_home()
            ctrl._send_command("G90", wait_for_ok=True)

            for cycle in range(1, loops + 1):
                gx = distance if axis == "x" else 0.0
                gy = distance if axis == "y" else 0.0
                gz = distance if axis == "z" else 0.0

                ok, resp = ctrl._send_command(f"G1 X{gx:.3f} Y{gy:.3f} Z{gz:.3f} F{feed:.0f}", wait_for_ok=True)
                if not ok:
                    print(f"  {label} move FAILED: {resp}")
                    break
                print(f"  [{label} {cycle}] -> {distance:.0f}mm ...", end="", flush=True)
                wait_idle(ctrl)
                s = ctrl.get_status()
                print(f" done  pos=({s.position_x:.1f}, {s.position_y:.1f})" if s else " done")
                time.sleep(0.3)

                ok, resp = ctrl._send_command(f"G1 X0 Y0 Z0 F{feed:.0f}", wait_for_ok=True)
                if not ok:
                    print(f"  {label} return FAILED: {resp}")
                    break
                print(f"  [{label} {cycle}] <- 0mm ...", end="", flush=True)
                wait_idle(ctrl)
                s = ctrl.get_status()
                print(f" done  pos=({s.position_x:.1f}, {s.position_y:.1f})" if s else " done")
                time.sleep(0.3)

            print(f"  {label} done.\n")
        print("All axes tested!")
    except KeyboardInterrupt:
        ctrl.stop()
        print("\nStopped.")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Motor / limit test (GRBL 1.1)")
    p.add_argument("--port", type=str, default=None)
    p.add_argument("--distance", type=float, default=40.0, help="Travel per leg (mm)")
    p.add_argument("--feed", type=float, default=900.0, help="Feed rate (mm/min)")
    p.add_argument("--axis", choices=["x", "y", "z", "xy", "all"], default="all")
    p.add_argument("--loops", type=int, default=1)
    p.add_argument("--skip-safety", action="store_true")
    p.add_argument("--seek-limits", action="store_true", help="Diagonal limit-seek test")
    p.add_argument("--step", type=float, default=2.0, help="Step size for limit seek (mm)")
    p.add_argument("--max-dist", type=float, default=500.0, help="Max travel before giving up (mm)")
    args = p.parse_args()

    if not args.skip_safety:
        print("=" * 55)
        print("  SAFETY CHECK")
        print("=" * 55)
        print("  1. Driver orientation correct?")
        print("  2. Vref ~0.6V?")
        print("  3. 24V power connected?")
        print("  4. Travel path clear?")
        print("=" * 55)
        try:
            if input("All good? (y): ").strip().lower() != "y":
                return
        except (EOFError, KeyboardInterrupt):
            return

    ctrl = GRBLController(port=args.port)
    if not ctrl.connect():
        print("Failed to connect.")
        return

    try:
        ctrl._send_command("$X")
        apply_settings(ctrl)
        ctrl.soft_home()
        ctrl._send_command("G90", wait_for_ok=True)

        if args.seek_limits:
            run_seek_limits(ctrl, max(0.2, args.step), args.feed, args.max_dist)
        else:
            axes_map = {"all": ["x", "y", "z"], "xy": ["x", "y"]}
            axes = axes_map.get(args.axis, [args.axis])
            run_axis_test(ctrl, axes, args.distance, args.feed, args.loops or 1)
    finally:
        ctrl.disconnect()


if __name__ == "__main__":
    main()
