"""
Limit-switch diagnostic for GRBL 1.1 + CNC Shield V3.

Shows the RAW pin state GRBL reports so you can verify switches work.
No baseline filtering — every pin that GRBL sees is printed so you can
tell exactly what changes (appears OR disappears) when you press a switch.

Hardware reality (CNC Shield V3 / Arduino Uno):
  - X-/X+ share one Arduino pin  (pin 9)
  - Y-/Y+ share one Arduino pin  (pin 10)
  - Z-/Z+ share one Arduino pin  (pin 11)
  Each pair is the SAME electrical input; GRBL can't tell + from -.

Bring-up sequence (per GRBL docs):
  1. $21=0  (hard limits OFF while testing)
  2. $22=1  (homing enabled for later)
  3. Observe raw pin states, toggle $5 to find correct polarity
  4. Press each switch → verify GRBL sees the change
  5. After confirmed: $21=1 (hard limits ON), then $H to home

Usage:
    python3 switch_diag.py --port /dev/cu.usbmodem14101
"""

import argparse
import re
import select
import sys
import time

import serial

# GRBL Pn: letters → human-readable names for your wiring
PIN_NAMES = {"X": "X", "Y": "Y+", "Z": "Y-"}


# ── Serial helpers ─────────────────────────────────────────────────────

def send(ser, cmd):
    """Send a command, wait briefly, return response text."""
    ser.reset_input_buffer()
    ser.write((cmd.strip() + "\n").encode())
    time.sleep(0.3)
    lines = []
    while ser.in_waiting:
        lines.append(ser.readline().decode("utf-8", errors="replace").strip())
    return " ".join(lines).strip()


def query_status(ser, timeout=0.5):
    """Send '?' and return the raw <…> status line."""
    ser.reset_input_buffer()
    ser.write(b"?")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if line.startswith("<") and ">" in line:
                return line
        else:
            time.sleep(0.01)
    return ""


def parse_state(status_line):
    """Extract GRBL state word (Idle, Alarm, Run …) from status line."""
    inner = status_line.strip("<>")
    return inner.split("|")[0] if inner else "?"


def parse_pins(status_line):
    """Extract active-pin letters from Pn: field (e.g. 'XYZ', 'Z', '')."""
    m = re.search(r"Pn:([A-Za-z]+)", status_line)
    return m.group(1) if m else ""


def pins_to_names(pins):
    """Pin letters → human names string, or '(none)'."""
    return ", ".join(PIN_NAMES.get(p, p) for p in pins) if pins else "(none)"


def sample_pins(ser, count=10):
    """Sample pins several times, return the most-common reading."""
    from collections import Counter
    readings = []
    for _ in range(count):
        raw = query_status(ser)
        readings.append(parse_pins(raw))
        time.sleep(0.1)
    if not readings:
        return ""
    return Counter(readings).most_common(1)[0][0]


# ── Main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Limit-switch diagnostic (GRBL 1.1 / CNC Shield V3)")
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)
    while ser.in_waiting:
        ser.readline()

    # ── Step 1: safe starting state ────────────────────────────────
    send(ser, "$X")       # clear any alarm
    send(ser, "$21=0")    # hard limits OFF (prevent alarm on switch press)
    send(ser, "$22=1")    # homing cycle enabled (for later)
    send(ser, "$X")       # clear alarm again in case $22 triggered one

    # ── Step 2: probe both $5 polarities ───────────────────────────
    print("\n" + "=" * 60)
    print("  PROBING LIMIT PIN POLARITY")
    print("  Do NOT press any switches right now.")
    print("=" * 60)
    time.sleep(1)

    send(ser, "$5=0"); send(ser, "$X"); time.sleep(0.3)
    pins_0 = sample_pins(ser, 10)

    send(ser, "$5=1"); send(ser, "$X"); time.sleep(0.3)
    pins_1 = sample_pins(ser, 10)

    print(f"\n  $5=0 → idle Pn: {pins_to_names(pins_0):20s}  (raw: '{pins_0}')")
    print(f"  $5=1 → idle Pn: {pins_to_names(pins_1):20s}  (raw: '{pins_1}')")
    print()
    print("  Correct polarity = the $5 where idle shows FEWER pins.")
    print("  (Pins visible at idle with no switches pressed are phantoms")
    print("   from floating/pulled-up inputs with no switch attached.)")

    # Pick best $5 — fewer idle phantoms wins; tie-break $5=1 (NO + pull-up)
    if len(pins_1) < len(pins_0):
        best_5 = "1"
    elif len(pins_0) < len(pins_1):
        best_5 = "0"
    else:
        best_5 = "1"
    idle_pins = pins_0 if best_5 == "0" else pins_1
    print(f"\n  → Using $5={best_5}  (idle pins: {pins_to_names(idle_pins)})")
    send(ser, f"$5={best_5}")
    send(ser, "$X")

    # ── Step 3: live polling — show RAW state ──────────────────────
    print("\n" + "=" * 60)
    print("  LIVE PIN MONITOR  ($21=0, hard limits OFF)")
    print("=" * 60)
    print(f"  $5={best_5}")
    print()
    print("  Shows RAW Pn: field from GRBL — no filtering.")
    print("  When you press a switch, pins will APPEAR or DISAPPEAR:")
    print("    NO switch + $5=0 → pin DISAPPEARS on press")
    print("    NO switch + $5=1 → pin APPEARS on press")
    print()
    print("  If NOTHING changes when you press → wiring/contact issue.")
    print()
    print("  Keys: 'i'+Enter = toggle $5  |  'r'+Enter = dump raw status")
    print("         Ctrl+C = quit")
    print("=" * 60)
    print()

    current_5 = best_5
    prev_pins = None
    n = 0

    try:
        while True:
            raw = query_status(ser)
            state = parse_state(raw)
            pins = parse_pins(raw)
            n += 1

            # Auto-unlock alarms (shouldn't happen with $21=0, but just in case)
            if "alarm" in state.lower():
                send(ser, "$X")

            # ── Change detection (raw — no baseline filtering) ─────
            change_str = ""
            if prev_pins is not None:
                appeared = set(pins) - set(prev_pins)
                vanished = set(prev_pins) - set(pins)
                if appeared:
                    names = ", ".join(PIN_NAMES.get(p, p) for p in sorted(appeared))
                    change_str += f"  ++ {names} APPEARED"
                if vanished:
                    names = ", ".join(PIN_NAMES.get(p, p) for p in sorted(vanished))
                    change_str += f"  -- {names} GONE"

            state_tag = f"  [{state}]" if state.lower() != "idle" else ""
            pin_str = pins_to_names(pins)
            print(f"  [{n:4d}] Pn: {pin_str:20s}{change_str}{state_tag}")

            prev_pins = pins

            # ── Keyboard commands ──────────────────────────────────
            if select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == "i":
                    new = "0" if current_5 == "1" else "1"
                    send(ser, f"$5={new}")
                    send(ser, "$X")
                    current_5 = new
                    prev_pins = None  # reset change tracking
                    print(f"\n  >> $5 toggled to {new}\n")
                elif cmd == "r":
                    # Dump full raw status for debugging
                    raw2 = query_status(ser)
                    print(f"\n  RAW: {raw2}\n")

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n\nDone.")
    finally:
        # Restore hard limits before exiting
        send(ser, "$21=1")
        send(ser, "$X")
        print("  ($21=1 restored — hard limits back ON)")
        ser.close()


if __name__ == "__main__":
    main()
