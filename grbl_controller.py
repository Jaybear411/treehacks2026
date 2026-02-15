"""
GRBL 1.1 Controller — serial communication with Arduino running GRBL 1.1.

Handles:
- Connection to Arduino via USB serial
- Sending G-code commands (absolute, relative, rapid)
- Real-time status queries (GRBL 1.1 pipe-delimited format)
- Limit switch detection and directional lockout
- Dual-Y axis: mirrors Y to Z (second Y motor on Z driver slot)
- Magnet engage/disengage on auxiliary axis
"""

import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Callable

import serial
from serial.tools import list_ports

from config import GRBL, GANTRY


class GRBLState(Enum):
    IDLE = "Idle"
    RUN = "Run"
    HOLD = "Hold"
    JOG = "Jog"
    ALARM = "Alarm"
    DOOR = "Door"
    CHECK = "Check"
    HOME = "Home"
    SLEEP = "Sleep"
    UNKNOWN = "Unknown"


@dataclass
class MachineStatus:
    state: GRBLState
    position_x: float
    position_y: float
    active_pins: str = ""


class GRBLController:
    """Controller for GRBL 1.1 based CNC/gantry systems."""

    def __init__(self, port: str = None, baud_rate: int = None):
        self.port = port or GRBL.port
        self.baud_rate = baud_rate or GRBL.baud_rate
        self.serial: Optional[serial.Serial] = None
        self.connected = False
        self.mirror_y_to_z: bool = GRBL.mirror_y_to_z

        self._position = (0.0, 0.0)
        self._state = GRBLState.UNKNOWN

        # Limit switch lockout flags
        # Wiring: GRBL Y pin → Y+, GRBL Z pin → Y-, GRBL X pin → X (direction inferred)
        self._active_limit_pins: str = ""
        self._limit_block_x_neg = False
        self._limit_block_x_pos = False
        self._limit_block_y_neg = False
        self._limit_block_y_pos = False
        self._last_x_dir = 0
        self._last_y_dir = 0

        # Phantom pin baseline — pins that read active at idle due to
        # floating/pull-up issues.  Populated once after connect so that
        # _update_limit_state() can ignore them.
        self._phantom_baseline: str = ""

        # Magnet state
        self._magnet_engaged = True

        self._lock = threading.RLock()
        self.on_status_update: Optional[Callable[[MachineStatus], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    # ── Port discovery ────────────────────────────────────────────────

    @staticmethod
    def list_ports() -> list:
        return [(p.device, p.description) for p in list_ports.comports()]

    @staticmethod
    def find_arduino() -> Optional[str]:
        for port, desc in GRBLController.list_ports():
            # Check both the port path AND description for known Arduino identifiers
            haystack = f"{port} {desc}".lower()
            if any(k in haystack for k in ['arduino', 'usbmodem', 'usbserial', 'ch340', 'ftdi', 'iousbhostdevice']):
                return port
        return None

    # ── Connection ────────────────────────────────────────────────────

    def connect(self, timeout: float = None) -> bool:
        timeout = timeout or GRBL.timeout

        # If the configured port doesn't exist on disk, auto-detect.
        if self.port:
            import os as _os
            if not _os.path.exists(self.port):
                print(f"Configured port {self.port} not found, auto-detecting...")
                self.port = self.find_arduino()

        if self.port is None:
            self.port = self.find_arduino()
            if self.port is None:
                print("ERROR: Could not auto-detect Arduino. Available ports:")
                for port, desc in self.list_ports():
                    print(f"  {port}: {desc}")
                return False
        try:
            self.serial = serial.Serial(port=self.port, baudrate=self.baud_rate, timeout=timeout)
            time.sleep(2.0)
            self._flush()
            # Wake GRBL
            self.serial.write(b"\r\n")
            time.sleep(0.1)
            self._flush()
            # Verify with status query
            self.serial.write(b"?")
            resp = self._read_response(timeout=2.0)
            if resp and "<" in resp:
                self.connected = True
                print(f"Connected to GRBL 1.1 on {self.port}")
                self._send_command("$X")          # clear any stale alarm
                self._apply_limit_settings()
                self._send_command("G90")
                self._sample_phantom_baseline()
                return True
            print(f"ERROR: No valid GRBL response. Got: {resp}")
            self.serial.close()
            return False
        except serial.SerialException as e:
            print(f"ERROR: Could not connect to {self.port}: {e}")
            return False

    def disconnect(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.connected = False
        print("Disconnected from GRBL")

    # ── GRBL limit / homing settings ──────────────────────────────────

    def _apply_limit_settings(self):
        """Apply essential GRBL settings on every connect.

        Merges the limit/homing settings with the machine rate and
        acceleration values proven in goto.py / home.py.

        $5=1           Limit pin invert (NO switches + pull-ups).
        $21=0 or 1     Hard limits — controlled by GRBL.enable_hard_limits.
        $22=1          Homing cycle enabled (allows $H).
        $23=3          Homing direction invert mask (seek +X, +Y).
        $24=25         Homing feed rate (mm/min).
        $25=500        Homing seek rate (mm/min).
        $27=1.0        Homing pull-off (mm).
        $110-$112      Max rate per axis (from GRBL.max_rate).
        $120-$122      Acceleration per axis (from GRBL.acceleration).
        """
        hl = "1" if GRBL.enable_hard_limits else "0"
        mr = f"{GRBL.max_rate:.0f}"
        ac = f"{GRBL.acceleration:.0f}"

        settings = [
            ("$5",   "1",   "limit pin invert (NO switches + pull-ups)"),
            ("$21",  hl,    f"hard limits {'ON' if GRBL.enable_hard_limits else 'OFF'}"),
            ("$22",  "1",   "homing cycle enabled"),
            ("$23",  "3",   "homing dir invert mask (seek +X, +Y)"),
            ("$24",  "25",  "homing feed (mm/min)"),
            ("$25",  "500", "homing seek (mm/min)"),
            ("$27",  "1.0", "homing pull-off (mm)"),
            ("$110", mr,    f"X max rate {mr} mm/min"),
            ("$111", mr,    f"Y max rate {mr} mm/min"),
            ("$112", mr,    f"Z max rate {mr} mm/min"),
            ("$120", ac,    f"X accel {ac} mm/s²"),
            ("$121", ac,    f"Y accel {ac} mm/s²"),
            ("$122", ac,    f"Z accel {ac} mm/s²"),
        ]
        print("Applying GRBL settings...")
        for key, val, desc in settings:
            ok, resp = self._send_command(f"{key}={val}")
            tag = "ok" if ok else resp.strip()
            print(f"  {key}={val:5s} ({desc}) -> {tag}")

    # ── Low-level serial ──────────────────────────────────────────────

    def _flush(self):
        if self.serial:
            self.serial.reset_input_buffer()

    def _read_response(self, timeout: float = None) -> str:
        timeout = timeout or GRBL.timeout
        if not self.serial:
            return ""
        lines = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.serial.in_waiting:
                line = self.serial.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    lines.append(line)
                    low = line.lower()
                    if low == "ok" or low.startswith("error") or low.startswith("alarm"):
                        break
                    if line.startswith("<") and line.endswith(">"):
                        break
            else:
                time.sleep(0.01)
        return "\n".join(lines)

    def _send_command(self, cmd: str, wait_for_ok: bool = True) -> Tuple[bool, str]:
        if not self.connected or not self.serial:
            return False, "Not connected"
        with self._lock:
            try:
                self.serial.write((cmd.strip() + "\n").encode("utf-8"))
                if not wait_for_ok:
                    return True, ""
                resp = self._read_response(timeout=GRBL.ack_timeout)
                low = resp.lower()
                ok = "ok" in low or resp.startswith("<")
                if "error" in low or "alarm" in low:
                    ok = False
                    if "alarm:1" in low:
                        self._mark_limit_hit_from_last_direction()
                    if self.on_error:
                        self.on_error(resp)
                self.get_status()
                return ok, resp
            except serial.SerialException as e:
                return False, str(e)

    # ── Status query (GRBL 1.1 format) ───────────────────────────────

    def get_status(self) -> Optional[MachineStatus]:
        """Query machine status. GRBL 1.1: <State|MPos:x,y,z|...|Pn:XYZ>"""
        if not self.connected:
            return None
        with self._lock:
            try:
                for _ in range(3):
                    self.serial.write(b"?")
                    resp = self._read_response(timeout=0.5)
                    # Find the status line
                    status_line = ""
                    for line in resp.splitlines():
                        if line.startswith("<") and ">" in line:
                            status_line = line
                            break
                    if not status_line:
                        continue

                    inner = status_line[1:status_line.index(">")]
                    fields = inner.split("|")

                    # Field 0: state (GRBL may report "Hold:0", "Alarm:1", etc.)
                    raw_state = fields[0].split(":")[0]
                    try:
                        state = GRBLState(raw_state)
                    except ValueError:
                        state = GRBLState.UNKNOWN

                    # Parse remaining fields
                    pos_x, pos_y = 0.0, 0.0
                    active_pins = ""
                    for field in fields[1:]:
                        if field.startswith("MPos:") or field.startswith("WPos:"):
                            coords = field.split(":")[1].split(",")
                            pos_x = float(coords[0])
                            pos_y = float(coords[1]) if len(coords) > 1 else 0.0
                        elif field.startswith("Pn:"):
                            active_pins = field[3:]

                    # Negate back to logical space when GRBL reports in
                    # negative work-area coordinates.
                    if GRBL.negate_grbl_coords:
                        pos_x = -pos_x
                        pos_y = -pos_y

                    self._position = (pos_x, pos_y)
                    self._state = state
                    self._update_limit_state(active_pins)
                    pos_x, pos_y = self._position

                    status = MachineStatus(
                        state=state,
                        position_x=pos_x,
                        position_y=pos_y,
                        active_pins=active_pins,
                    )
                    if self.on_status_update:
                        self.on_status_update(status)
                    return status
            except Exception as e:
                print(f"Status query error: {e}")
        return None

    # ── Homing / zero ─────────────────────────────────────────────────

    def home(self) -> bool:
        """Run GRBL homing cycle ($H). Requires limit switches & $22=1."""
        print("Running homing cycle ($H)...")
        ok, resp = self._send_command("$H")
        if ok:
            self._position = (0.0, 0.0)
            print("Homing complete")
        else:
            print(f"Homing failed: {resp}")
        return ok

    def home_to_origin(self, seek_feed: float = 3500, fine_feed: float = 800,
                       fine_step: float = 0.5,
                       backoff: float = 3.0, pulloff: float = 1.5,
                       max_travel: float = 450.0) -> bool:
        """Smooth two-phase switch-seek homing to front-left origin.

        Phase 1 is one fluid continuous G1 move (not stepped) — polls the
        status register for pin/alarm and feed-holds on contact.
        Phase 2 is a short stepped approach for precision.

        Mirrors the proven home.py behaviour.
        """
        if not self.connected:
            return False

        # Temporarily disable hard limits & clear lockouts
        self._send_command("$21=0")
        self.unlock()
        self._send_command("G90")
        self.soft_home()  # temporary zero

        # ── Continuous seek (smooth single move) ───────────────────
        def _seek_continuous(pin: str, dx: float, dy: float,
                             feed: float, max_mm: float, label: str) -> bool:
            """One smooth G1 move; polls for switch pin or alarm."""
            x_dist = dx * max_mm
            y_dist = dy * max_mm
            z_dist = (y_dist * GRBL.y_z_scale
                      if self.mirror_y_to_z and abs(y_dist) > 1e-6 else 0)
            parts = []
            if abs(x_dist) > 1e-6:
                parts.append(f"X{x_dist:.1f}")
            if abs(y_dist) > 1e-6:
                parts.append(f"Y{y_dist:.1f}")
            if abs(z_dist) > 1e-6:
                parts.append(f"Z{z_dist:.1f}")

            self._send_command("G91")
            self._send_command(f"G1 {' '.join(parts)} F{feed:.0f}")
            time.sleep(0.3)  # let GRBL transition Idle → Run

            hit = False
            saw_run = False
            idle_count = 0

            while True:
                status = self.get_status()
                if not status:
                    time.sleep(0.03)
                    continue

                if status.state in (GRBLState.RUN,):
                    saw_run = True
                    idle_count = 0

                # Switch hit via pin
                if pin in status.active_pins:
                    hit = True
                    self.serial.write(b"!")  # feed hold — immediate stop
                    print(f"    {label} switch HIT (pin)")
                    break

                # Switch hit via alarm
                if status.state == GRBLState.ALARM:
                    hit = True
                    print(f"    {label} switch HIT (alarm)")
                    break

                # Move finished without hitting switch
                if status.state == GRBLState.IDLE and saw_run:
                    idle_count += 1
                    if idle_count >= 3:
                        break

                time.sleep(0.03)

            # Wait for full stop
            for _ in range(100):
                status = self.get_status()
                if status and status.state in (GRBLState.IDLE, GRBLState.HOLD):
                    break
                if status and status.state == GRBLState.ALARM:
                    break
                time.sleep(0.05)

            # Soft-reset to clear any remaining queued motion
            self.serial.write(b"\x18")
            time.sleep(1.0)
            self._flush()
            self.unlock()
            self._send_command("G90")
            return hit

        # ── Stepped seek (short distance, precision) ──────────────
        def _seek_stepped(pin: str, dx: float, dy: float,
                          step: float, feed: float,
                          max_mm: float, label: str) -> bool:
            traveled = 0.0
            self._send_command("G91")
            while traveled < max_mm:
                status = self.get_status()
                if status and pin in status.active_pins:
                    self._send_command("G90")
                    return True
                if status and status.state == GRBLState.ALARM:
                    self.unlock()
                    self._send_command("G91")
                z_part = (f" Z{dy * step * GRBL.y_z_scale:.3f}"
                          if self.mirror_y_to_z and abs(dy) > 1e-6 else "")
                self._send_command(
                    f"G1 X{dx * step:.3f} Y{dy * step:.3f}{z_part} F{feed:.0f}")
                self._wait_for_idle(timeout=5)
                traveled += step
            self._send_command("G90")
            return False

        # ── Home one axis ─────────────────────────────────────────
        def _home_axis(pin: str, name: str, dx: float, dy: float) -> bool:
            # Phase 1: smooth continuous seek
            print(f"  [{name}] Seeking switch (F{seek_feed})...")
            hit_dx, hit_dy = dx, dy
            found = _seek_continuous(pin, dx, dy, seek_feed, max_travel, name)
            if not found:
                print(f"  [{name}] Trying opposite direction...")
                found = _seek_continuous(pin, -dx, -dy, seek_feed, max_travel, name)
                if found:
                    hit_dx, hit_dy = -dx, -dy
            if not found:
                print(f"  [{name}] Switch not found!")
                return False

            # Back off (single smooth move)
            print(f"  [{name}] Backing off {backoff}mm...")
            self._send_command("G91")
            z_part = (f" Z{-hit_dy * backoff * GRBL.y_z_scale:.3f}"
                      if self.mirror_y_to_z and abs(hit_dy) > 1e-6 else "")
            self._send_command(
                f"G1 X{-hit_dx * backoff:.3f} Y{-hit_dy * backoff:.3f}"
                f"{z_part} F{seek_feed:.0f}")
            self._wait_for_idle(timeout=10)
            self._send_command("G90")

            # Phase 2: slow stepped approach (short distance, precision)
            print(f"  [{name}] Slow approach (F{fine_feed})...")
            found = _seek_stepped(pin, hit_dx, hit_dy, fine_step, fine_feed,
                                  backoff + 5.0, name)
            if not found:
                print(f"  [{name}] WARNING: switch not hit on slow approach")
                return False

            # Final pull-off (single smooth move)
            print(f"  [{name}] Pull-off {pulloff}mm...")
            self._send_command("G91")
            z_part = (f" Z{-hit_dy * pulloff * GRBL.y_z_scale:.3f}"
                      if self.mirror_y_to_z and abs(hit_dy) > 1e-6 else "")
            self._send_command(
                f"G1 X{-hit_dx * pulloff:.3f} Y{-hit_dy * pulloff:.3f}"
                f"{z_part} F{fine_feed:.0f}")
            self._wait_for_idle(timeout=10)
            self._send_command("G90")

            print(f"  [{name}] Done.")
            return True

        # ── Execute ───────────────────────────────────────────────
        print("\n=== HOMING TO FRONT-LEFT ORIGIN ===")
        y_ok = _home_axis("Y", "Y-front", dx=0, dy=1)
        x_ok = _home_axis("X", "X-left",  dx=1, dy=0)

        # Set origin
        self.soft_home()

        # Restore hard limits to configured state
        self._send_command(f"$21={'1' if GRBL.enable_hard_limits else '0'}")
        self.unlock()
        self._limit_block_x_neg = False
        self._limit_block_x_pos = False
        self._limit_block_y_neg = False
        self._limit_block_y_pos = False

        if x_ok and y_ok:
            print("=== HOMED — Origin (0, 0) set ===\n")
        else:
            print("=== PARTIAL HOME — Origin set at current position ===\n")
        return x_ok and y_ok

    def soft_home(self) -> bool:
        """Set current position as (0,0) without moving."""
        print("Setting current position as home (0,0)")
        cmd = "G92 X0 Y0 Z0" if self.mirror_y_to_z else "G92 X0 Y0"
        ok, _ = self._send_command(cmd)
        if ok:
            self._position = (0.0, 0.0)
        return ok

    # ── Movement ──────────────────────────────────────────────────────

    def move_to(self, x: float, y: float, feed_rate: float = None, wait: bool = True) -> bool:
        """Absolute move to (x, y) mm. Moves X then Y so a limit hit stops only that axis and the other still runs."""
        feed_rate = feed_rate or GRBL.feed_rate
        x = max(0.0, min(GANTRY.width_mm, x))
        y = max(0.0, min(GANTRY.height_mm, y))
        # If already in ALARM (e.g. from a previous jog into limit), recover so we can move other axes
        status = self.get_status()
        if status and status.state == GRBLState.ALARM:
            self.unlock()
        cx, cy = self._position[0], self._position[1]
        eps = 0.02

        # Move X first
        if abs(x - cx) > eps:
            dx = x - cx
            if self._movement_allowed(dx, 0):
                self._remember_direction(dx, 0)
                gx = -x if GRBL.negate_grbl_coords else x
                ok, _ = self._send_command(f"G1 X{gx:.3f} F{feed_rate:.0f}")
                if ok and wait:
                    if not self._wait_for_idle_or_alarm():
                        self._recover_after_move()
                    self.get_status()  # update position (and limit state)
            elif wait:
                self.get_status()
        elif wait:
            self.get_status()

        # Move Y (and Z if dual-Y)
        cy_now = self._position[1]
        if abs(y - cy_now) > eps:
            dy = y - cy_now
            if self._movement_allowed(0, dy):
                self._remember_direction(0, dy)
                gy = -y if GRBL.negate_grbl_coords else y
                z_part = f" Z{gy * GRBL.y_z_scale:.3f}" if self.mirror_y_to_z else ""
                ok, _ = self._send_command(f"G1 Y{gy:.3f}{z_part} F{feed_rate:.0f}")
                if ok and wait:
                    if not self._wait_for_idle_or_alarm():
                        self._recover_after_move()
                    self.get_status()
            elif wait:
                self.get_status()
        elif wait:
            self.get_status()

        return True

    def rapid_to(self, x: float, y: float, wait: bool = True) -> bool:
        """Rapid (G0) move to (x, y). Moves X then Y so a limit hit stops only that axis and the other still runs."""
        x = max(0.0, min(GANTRY.width_mm, x))
        y = max(0.0, min(GANTRY.height_mm, y))
        status = self.get_status()
        if status and status.state == GRBLState.ALARM:
            self.unlock()
        cx, cy = self._position[0], self._position[1]
        eps = 0.02

        # Move X first
        if abs(x - cx) > eps:
            dx = x - cx
            if self._movement_allowed(dx, 0):
                self._remember_direction(dx, 0)
                gx = -x if GRBL.negate_grbl_coords else x
                ok, _ = self._send_command(f"G0 X{gx:.3f}")
                if ok and wait:
                    if not self._wait_for_idle_or_alarm():
                        self._recover_after_move()
                    self.get_status()
            elif wait:
                self.get_status()
        elif wait:
            self.get_status()

        # Move Y (and Z if dual-Y)
        cy_now = self._position[1]
        if abs(y - cy_now) > eps:
            dy = y - cy_now
            if self._movement_allowed(0, dy):
                self._remember_direction(0, dy)
                gy = -y if GRBL.negate_grbl_coords else y
                z_part = f" Z{gy * GRBL.y_z_scale:.3f}" if self.mirror_y_to_z else ""
                ok, _ = self._send_command(f"G0 Y{gy:.3f}{z_part}")
                if ok and wait:
                    if not self._wait_for_idle_or_alarm():
                        self._recover_after_move()
                    self.get_status()
            elif wait:
                self.get_status()
        elif wait:
            self.get_status()

        return True

    def jog(self, dx: float, dy: float, feed_rate: float = None) -> bool:
        """Relative move by (dx, dy) mm. Moves X then Y; a limit hit stops only that axis."""
        feed_rate = feed_rate or GRBL.feed_rate
        new_x = max(0.0, min(GANTRY.width_mm, self._position[0] + dx))
        new_y = max(0.0, min(GANTRY.height_mm, self._position[1] + dy))
        dx = new_x - self._position[0]
        dy = new_y - self._position[1]
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return True
        self._send_command("G91", wait_for_ok=True)
        try:
            if abs(dx) >= 0.01 and self._movement_allowed(dx, 0):
                self._remember_direction(dx, 0)
                gdx = -dx if GRBL.negate_grbl_coords else dx
                ok, _ = self._send_command(f"G1 X{gdx:.3f} F{feed_rate:.0f}")
                if ok:
                    if not self._wait_for_idle_or_alarm():
                        self._recover_after_move()
                    self.get_status()
            if abs(dy) >= 0.01 and self._movement_allowed(0, dy):
                self._remember_direction(0, dy)
                gdy = -dy if GRBL.negate_grbl_coords else dy
                z_part = f" Z{gdy * GRBL.y_z_scale:.3f}" if self.mirror_y_to_z else ""
                ok, _ = self._send_command(f"G1 Y{gdy:.3f}{z_part} F{feed_rate:.0f}")
                if ok:
                    if not self._wait_for_idle_or_alarm():
                        self._recover_after_move()
                    self.get_status()
        finally:
            self._send_command("G90", wait_for_ok=True)
        return True

    # ── Limit switch logic ────────────────────────────────────────────

    def _remember_direction(self, dx: float, dy: float):
        self._last_x_dir = 1 if dx > 0 else (-1 if dx < 0 else 0)
        self._last_y_dir = 1 if dy > 0 else (-1 if dy < 0 else 0)

    def _movement_allowed(self, dx: float, dy: float) -> bool:
        if dx < 0 and self._limit_block_x_neg:
            print("Blocked: X negative limit active")
            return False
        if dx > 0 and self._limit_block_x_pos:
            print("Blocked: X positive limit active")
            return False
        if dy < 0 and self._limit_block_y_neg:
            print("Blocked: Y negative limit active")
            return False
        if dy > 0 and self._limit_block_y_pos:
            print("Blocked: Y positive limit active")
            return False
        return True

    def _sample_phantom_baseline(self, samples: int = 5):
        """Sample idle pin readings so we can ignore phantom (floating) pins.

        Must be called right after connect, BEFORE any movement, with no
        switches physically pressed.
        """
        from collections import Counter
        readings: list[str] = []
        for _ in range(samples):
            status = self.get_status()
            if status:
                readings.append(status.active_pins)
            time.sleep(0.15)
        if readings:
            most_common = Counter(readings).most_common(1)[0][0]
        else:
            most_common = ""
        self._phantom_baseline = most_common
        if most_common:
            pin_names = {"X": "X", "Y": "Y+", "Z": "Y-"}
            names = ", ".join(pin_names.get(p, p) for p in most_common)
            print(f"Phantom baseline pins (ignored): {names}")

    def _update_limit_state(self, active_pins: str):
        """
        Update limit lockouts from GRBL 1.1 Pn: field.
        Wiring: Y pin → Y+, Z pin → Y-, X pin → inferred from last direction.

        Pins present in _phantom_baseline are ignored (they read active even
        at idle due to floating / pull-up issues and are not real switch hits).
        All three axes now use direction-aware logic: a pin only blocks
        movement in the direction the gantry was last traveling.
        """
        self._active_limit_pins = active_pins

        # Filter out phantom pins that were active at idle
        real_pins = "".join(
            p for p in active_pins if p not in self._phantom_baseline
        )

        x_active = "X" in real_pins
        y_pos_active = "Y" in real_pins   # GRBL Y pin = Y+
        y_neg_active = "Z" in real_pins   # GRBL Z pin = Y-

        # ── X: direction-aware (unchanged) ──
        if x_active:
            if self._last_x_dir < 0:
                self._limit_block_x_neg = True
                self._limit_block_x_pos = False
            elif self._last_x_dir > 0:
                self._limit_block_x_pos = True
                self._limit_block_x_neg = False
        else:
            self._limit_block_x_neg = False
            self._limit_block_x_pos = False

        # ── Y+: direction-aware (was unconditional — caused phantom lockout) ──
        if y_pos_active:
            if self._last_y_dir > 0:
                self._limit_block_y_pos = True
            # Don't set if we weren't moving in Y+ direction
        else:
            self._limit_block_y_pos = False

        # ── Y-: direction-aware ──
        if y_neg_active:
            if self._last_y_dir < 0:
                self._limit_block_y_neg = True
            # Don't set if we weren't moving in Y- direction
        else:
            self._limit_block_y_neg = False

        # Snap coordinates to known limits
        x, y = self._position
        if self._limit_block_x_neg:
            x = GANTRY.min_x
        elif self._limit_block_x_pos:
            x = GANTRY.max_x
        if self._limit_block_y_neg:
            y = GANTRY.min_y
        elif self._limit_block_y_pos:
            y = GANTRY.max_y
        self._position = (x, y)

    def _mark_limit_hit_from_last_direction(self):
        """Fallback when ALARM:1 fires but no Pn: data available."""
        if self._last_x_dir < 0:
            self._limit_block_x_neg = True
            self._limit_block_x_pos = False
        elif self._last_x_dir > 0:
            self._limit_block_x_pos = True
            self._limit_block_x_neg = False
        if self._last_y_dir < 0:
            self._limit_block_y_neg = True
            self._limit_block_y_pos = False
        elif self._last_y_dir > 0:
            self._limit_block_y_pos = True
            self._limit_block_y_neg = False

        x, y = self._position
        if self._limit_block_x_neg:
            x = GANTRY.min_x
        elif self._limit_block_x_pos:
            x = GANTRY.max_x
        if self._limit_block_y_neg:
            y = GANTRY.min_y
        elif self._limit_block_y_pos:
            y = GANTRY.max_y
        self._position = (x, y)

    # ── Magnet engage/disengage ───────────────────────────────────────

    def _run_relative_axis_move(self, axis: str, delta: float, feed_rate: float) -> bool:
        axis = axis.strip().upper()
        if len(axis) != 1 or not axis.isalpha():
            print(f"Invalid magnet axis '{axis}'")
            return False
        if abs(delta) < 1e-9:
            return True
        if not self._send_command("G91", wait_for_ok=True)[0]:
            return False
        try:
            ok, _ = self._send_command(f"G1 {axis}{delta:.3f} F{feed_rate:.0f}", wait_for_ok=True)
        finally:
            self._send_command("G90", wait_for_ok=True)
        if ok:
            self._wait_for_idle()
        return ok

    def engage_magnet(self) -> bool:
        if not GRBL.magnet_lift_enabled or self._magnet_engaged:
            return True
        print("Magnet: engage")
        ok = self._run_relative_axis_move(GRBL.magnet_axis, GRBL.magnet_engage_delta_units, GRBL.magnet_feed_rate)
        if ok:
            self._magnet_engaged = True
            if GRBL.magnet_settle_sec > 0:
                time.sleep(GRBL.magnet_settle_sec)
        return ok

    def disengage_magnet(self) -> bool:
        if not GRBL.magnet_lift_enabled or not self._magnet_engaged:
            return True
        print("Magnet: disengage")
        ok = self._run_relative_axis_move(GRBL.magnet_axis, GRBL.magnet_disengage_delta_units, GRBL.magnet_feed_rate)
        if ok:
            self._magnet_engaged = False
            if GRBL.magnet_settle_sec > 0:
                time.sleep(GRBL.magnet_settle_sec)
        return ok

    # ── High-level operations ─────────────────────────────────────────

    def drag_object_to(self, from_x: float, from_y: float, to_x: float, to_y: float) -> bool:
        """Home → move to object → move to destination (front/pickup zone).

        Sequence:
          1. Go home (0, 0) so the gantry starts from a known position.
          2. Move to the object at (from_x, from_y).
          3. Move to the destination at (to_x, to_y) — typically the front pickup zone.

        All coordinates are clamped to [0, max_travel] before sending.
        """
        # Clamp inputs to physical boundaries
        from_x = max(0.0, min(GANTRY.width_mm, from_x))
        from_y = max(0.0, min(GANTRY.height_mm, from_y))
        to_x = max(0.0, min(GANTRY.width_mm, to_x))
        to_y = max(0.0, min(GANTRY.height_mm, to_y))

        # Step 1: Physical home using limit switches (establishes true 0,0)
        print("Homing to limit switches before fetch...")
        if not self.home_to_origin():
            print("WARNING: Limit-switch home failed, continuing anyway")

        # Step 2: Move to the object
        print(f"Moving to object at ({from_x:.1f}, {from_y:.1f}) mm")
        if not self.move_to(from_x, from_y):
            return False

        # Step 3: Move to the front / pickup zone
        print(f"Moving to front at ({to_x:.1f}, {to_y:.1f}) mm")
        ok = self.move_to(to_x, to_y, feed_rate=GRBL.drag_rate)
        return ok

    def go_to_pickup_zone(self) -> bool:
        return self.move_to(GANTRY.pickup_x_mm, GANTRY.pickup_y_mm)

    def go_home(self) -> bool:
        return self.rapid_to(GANTRY.home_x_mm, GANTRY.home_y_mm)

    def stop(self) -> bool:
        if self.serial:
            self.serial.write(b"!")
            return True
        return False

    def resume(self) -> bool:
        if self.serial:
            self.serial.write(b"~")
            return True
        return False

    def reset(self) -> bool:
        if self.serial:
            self.serial.write(b"\x18")
            time.sleep(1.0)
            self._flush()
            return True
        return False

    def unlock(self) -> bool:
        """Send $X to clear alarm lock and reset all limit lockout flags."""
        ok, _ = self._send_command("$X")
        if ok:
            self._limit_block_x_neg = False
            self._limit_block_x_pos = False
            self._limit_block_y_neg = False
            self._limit_block_y_pos = False
        return ok

    def _wait_for_idle(self, timeout: float = 60.0) -> bool:
        start = time.time()
        while (time.time() - start) < timeout:
            status = self.get_status()
            if status:
                if status.state == GRBLState.IDLE:
                    return True
                if status.state == GRBLState.ALARM:
                    return False
            time.sleep(0.1)
        return False

    def _wait_for_idle_or_alarm(self, timeout: float = 60.0) -> bool:
        """Wait until IDLE (True) or ALARM (False). On ALARM, caller should call _recover_after_move()."""
        start = time.time()
        while (time.time() - start) < timeout:
            status = self.get_status()
            if status:
                if status.state == GRBLState.IDLE:
                    return True
                if status.state == GRBLState.ALARM:
                    return False
            time.sleep(0.1)
        return False

    def _recover_after_move(self) -> None:
        """Update limit state from status; clear alarm ($X) so motion can continue in other directions."""
        status = self.get_status()
        if status and status.state == GRBLState.ALARM:
            self.unlock()

    # ── Properties ────────────────────────────────────────────────────

    @property
    def position(self) -> Tuple[float, float]:
        return self._position

    @property
    def state(self) -> GRBLState:
        return self._state

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


# ── Quick-test helpers ────────────────────────────────────────────────

def test_connection():
    print("Available serial ports:")
    for port, desc in GRBLController.list_ports():
        print(f"  {port}: {desc}")
    c = GRBLController()
    if c.connect():
        s = c.get_status()
        if s:
            print(f"State: {s.state.value}  Position: ({s.position_x}, {s.position_y})  Pins: {s.active_pins}")
        c.disconnect()
        return True
    return False


def test_movement():
    with GRBLController() as c:
        if not c.connected:
            return False
        c.soft_home()
        print("Moving to (50, 50)...")
        c.move_to(50, 50)
        print("Moving to (100, 100)...")
        c.move_to(100, 100)
        print("Returning home...")
        c.go_home()
        print("Test complete!")
        return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_movement()
    else:
        test_connection()
