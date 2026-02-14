"""
GRBL Controller - Serial communication with Arduino running GRBL firmware.

This module handles:
- Connection to Arduino via USB serial
- Sending G-code commands for movement
- Homing, jogging, and absolute positioning
- Status queries and error handling

GRBL Commands Reference:
- G0 X Y: Rapid move (fastest, no load)
- G1 X Y F: Linear move at feed rate F (mm/min)
- G90: Absolute positioning mode
- G91: Relative/incremental positioning mode
- G28: Return to home position
- $H: Homing cycle
- ?: Real-time status query
- !: Feed hold (pause)
- ~: Cycle start (resume)
- Ctrl+X: Soft reset
"""

import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Callable

import serial
from serial.tools import list_ports

from config import GRBL, GANTRY


class GRBLState(Enum):
    """GRBL machine states."""
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
    """Current machine status."""
    state: GRBLState
    position_x: float
    position_y: float
    feed_rate: float = 0.0
    buffer_available: int = 15


class GRBLController:
    """
    Controller for GRBL-based CNC/gantry systems.
    
    Usage:
        controller = GRBLController()
        controller.connect()
        controller.home()
        controller.move_to(100, 100)
        controller.disconnect()
    """
    
    def __init__(self, port: str = None, baud_rate: int = None):
        self.port = port or GRBL.port
        self.baud_rate = baud_rate or GRBL.baud_rate
        self.serial: Optional[serial.Serial] = None
        self.connected = False
        
        # Current known position
        self._position = (0.0, 0.0)
        self._state = GRBLState.UNKNOWN
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Callbacks
        self.on_status_update: Optional[Callable[[MachineStatus], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
    
    @staticmethod
    def list_ports() -> list:
        """List available serial ports."""
        ports = list_ports.comports()
        return [(p.device, p.description) for p in ports]
    
    @staticmethod
    def find_arduino() -> Optional[str]:
        """Try to auto-detect Arduino port."""
        for port, desc in GRBLController.list_ports():
            desc_lower = desc.lower()
            if any(x in desc_lower for x in ['arduino', 'usbmodem', 'usbserial', 'ch340', 'ftdi']):
                return port
        return None
    
    def connect(self, timeout: float = None) -> bool:
        """
        Connect to GRBL controller.
        
        Returns True if connection successful.
        """
        timeout = timeout or GRBL.timeout
        
        # Try auto-detect if port doesn't exist
        if self.port is None:
            self.port = self.find_arduino()
            if self.port is None:
                print("ERROR: Could not auto-detect Arduino. Available ports:")
                for port, desc in self.list_ports():
                    print(f"  {port}: {desc}")
                return False
        
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=timeout
            )
            
            # Wait for GRBL to initialize (it sends a welcome message)
            time.sleep(2.0)
            
            # Clear any startup messages
            self._flush_input()
            
            # Send a newline to wake up GRBL
            self.serial.write(b"\r\n")
            time.sleep(0.1)
            self._flush_input()
            
            # Verify connection with status query
            self.serial.write(b"?\n")
            response = self._read_response(timeout=2.0)
            
            if response and ('<' in response or 'ok' in response.lower()):
                self.connected = True
                print(f"Connected to GRBL on {self.port}")
                
                # Set absolute positioning mode
                self._send_command("G90")
                
                return True
            else:
                print(f"ERROR: No valid GRBL response. Got: {response}")
                self.serial.close()
                return False
                
        except serial.SerialException as e:
            print(f"ERROR: Could not connect to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from GRBL controller."""
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.connected = False
        print("Disconnected from GRBL")
    
    def _flush_input(self):
        """Clear input buffer."""
        if self.serial:
            self.serial.reset_input_buffer()
    
    def _read_response(self, timeout: float = None) -> str:
        """Read response from GRBL."""
        timeout = timeout or GRBL.timeout
        if not self.serial:
            return ""
        
        response_lines = []
        start = time.time()
        
        while (time.time() - start) < timeout:
            if self.serial.in_waiting:
                line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    response_lines.append(line)
                    # Check for completion markers
                    if line.lower() in ['ok', 'error']:
                        break
                    if line.startswith('<') and line.endswith('>'):
                        break
            else:
                time.sleep(0.01)
        
        return '\n'.join(response_lines)
    
    def _send_command(self, cmd: str, wait_for_ok: bool = True) -> Tuple[bool, str]:
        """
        Send a G-code command and wait for response.
        
        Returns (success, response_text)
        """
        if not self.connected or not self.serial:
            return False, "Not connected"
        
        with self._lock:
            try:
                # Ensure command ends with newline
                cmd = cmd.strip() + '\n'
                self.serial.write(cmd.encode('utf-8'))
                
                if wait_for_ok:
                    response = self._read_response(timeout=GRBL.ack_timeout)
                    success = 'ok' in response.lower() or response.startswith('<')
                    
                    if 'error' in response.lower():
                        success = False
                        if self.on_error:
                            self.on_error(response)
                    
                    return success, response
                else:
                    return True, ""
                    
            except serial.SerialException as e:
                return False, str(e)
    
    def get_status(self) -> Optional[MachineStatus]:
        """Query current machine status."""
        if not self.connected:
            return None
        
        with self._lock:
            try:
                self.serial.write(b"?\n")
                response = self._read_response(timeout=1.0)
                
                # Parse status response: <Idle|MPos:0.000,0.000,0.000|...>
                if response.startswith('<') and '>' in response:
                    # Extract state
                    inner = response[1:response.index('>')]
                    parts = inner.split('|')
                    
                    state_str = parts[0]
                    try:
                        state = GRBLState(state_str)
                    except ValueError:
                        state = GRBLState.UNKNOWN
                    
                    # Extract position
                    pos_x, pos_y = 0.0, 0.0
                    for part in parts:
                        if part.startswith('MPos:') or part.startswith('WPos:'):
                            coords = part.split(':')[1].split(',')
                            pos_x = float(coords[0])
                            pos_y = float(coords[1])
                            break
                    
                    self._position = (pos_x, pos_y)
                    self._state = state
                    
                    status = MachineStatus(
                        state=state,
                        position_x=pos_x,
                        position_y=pos_y
                    )
                    
                    if self.on_status_update:
                        self.on_status_update(status)
                    
                    return status
                    
            except Exception as e:
                print(f"Status query error: {e}")
        
        return None
    
    def home(self) -> bool:
        """
        Run homing cycle.
        
        NOTE: Requires limit switches configured in GRBL.
        If you don't have limit switches, use soft_home() instead.
        """
        print("Running homing cycle...")
        success, response = self._send_command("$H")
        if success:
            self._position = (0.0, 0.0)
            print("Homing complete")
        else:
            print(f"Homing failed: {response}")
        return success
    
    def soft_home(self) -> bool:
        """
        Set current position as home (0,0) without physical homing.
        
        Use this if you don't have limit switches.
        """
        print("Setting current position as home (0,0)")
        success, _ = self._send_command("G92 X0 Y0")
        if success:
            self._position = (0.0, 0.0)
        return success
    
    def move_to(self, x: float, y: float, feed_rate: float = None, 
                wait: bool = True) -> bool:
        """
        Move to absolute position (x, y) in mm.
        
        Args:
            x: Target X position in mm
            y: Target Y position in mm
            feed_rate: Speed in mm/min (default: GRBL.feed_rate)
            wait: If True, wait for move to complete
        """
        feed_rate = feed_rate or GRBL.feed_rate
        
        # Clamp to gantry limits
        x = max(GANTRY.min_x, min(GANTRY.max_x, x))
        y = max(GANTRY.min_y, min(GANTRY.max_y, y))
        
        cmd = f"G1 X{x:.3f} Y{y:.3f} F{feed_rate:.0f}"
        success, response = self._send_command(cmd)
        
        if success and wait:
            self._wait_for_idle()
            self._position = (x, y)
        
        return success
    
    def rapid_to(self, x: float, y: float, wait: bool = True) -> bool:
        """
        Rapid move to position (fastest, for positioning without load).
        """
        # Clamp to gantry limits
        x = max(GANTRY.min_x, min(GANTRY.max_x, x))
        y = max(GANTRY.min_y, min(GANTRY.max_y, y))
        
        cmd = f"G0 X{x:.3f} Y{y:.3f}"
        success, response = self._send_command(cmd)
        
        if success and wait:
            self._wait_for_idle()
            self._position = (x, y)
        
        return success
    
    def jog(self, dx: float, dy: float, feed_rate: float = None) -> bool:
        """
        Relative jog move by (dx, dy) mm.
        """
        feed_rate = feed_rate or GRBL.feed_rate
        
        # Switch to relative mode, move, switch back
        self._send_command("G91", wait_for_ok=True)
        cmd = f"G1 X{dx:.3f} Y{dy:.3f} F{feed_rate:.0f}"
        success, _ = self._send_command(cmd)
        self._send_command("G90", wait_for_ok=True)
        
        if success:
            self._wait_for_idle()
            self._position = (
                self._position[0] + dx,
                self._position[1] + dy
            )
        
        return success
    
    def drag_object_to(self, from_x: float, from_y: float, 
                       to_x: float, to_y: float) -> bool:
        """
        Move to object location, then drag it to destination.
        
        Uses slower speed for the dragging portion.
        """
        # First, rapid move to object location
        print(f"Moving to object at ({from_x:.1f}, {from_y:.1f})")
        if not self.rapid_to(from_x, from_y):
            return False
        
        # Brief pause to let magnet couple
        time.sleep(0.3)
        
        # Drag at slower speed
        print(f"Dragging to ({to_x:.1f}, {to_y:.1f})")
        return self.move_to(to_x, to_y, feed_rate=GRBL.drag_rate)
    
    def go_to_pickup_zone(self) -> bool:
        """Move to the pickup zone."""
        return self.move_to(GANTRY.pickup_x_mm, GANTRY.pickup_y_mm)
    
    def go_home(self) -> bool:
        """Move to home position."""
        return self.rapid_to(GANTRY.home_x_mm, GANTRY.home_y_mm)
    
    def stop(self) -> bool:
        """Emergency stop - feed hold."""
        if self.serial:
            self.serial.write(b"!")
            return True
        return False
    
    def resume(self) -> bool:
        """Resume after feed hold."""
        if self.serial:
            self.serial.write(b"~")
            return True
        return False
    
    def reset(self) -> bool:
        """Soft reset GRBL."""
        if self.serial:
            self.serial.write(b"\x18")  # Ctrl+X
            time.sleep(1.0)
            self._flush_input()
            return True
        return False
    
    def _wait_for_idle(self, timeout: float = 60.0) -> bool:
        """Wait for machine to reach idle state."""
        start = time.time()
        while (time.time() - start) < timeout:
            status = self.get_status()
            if status and status.state == GRBLState.IDLE:
                return True
            time.sleep(0.1)
        return False
    
    @property
    def position(self) -> Tuple[float, float]:
        """Current position (x, y) in mm."""
        return self._position
    
    @property
    def state(self) -> GRBLState:
        """Current machine state."""
        return self._state
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# Convenience functions for quick testing
def test_connection():
    """Test GRBL connection."""
    print("Available serial ports:")
    for port, desc in GRBLController.list_ports():
        print(f"  {port}: {desc}")
    
    print(f"\nAttempting connection to {GRBL.port}...")
    
    controller = GRBLController()
    if controller.connect():
        status = controller.get_status()
        if status:
            print(f"State: {status.state.value}")
            print(f"Position: ({status.position_x}, {status.position_y})")
        controller.disconnect()
        return True
    return False


def test_movement():
    """Test basic movement."""
    with GRBLController() as controller:
        if not controller.connected:
            return False
        
        print("Setting soft home...")
        controller.soft_home()
        
        print("Moving to (50, 50)...")
        controller.move_to(50, 50)
        
        print("Moving to (100, 100)...")
        controller.move_to(100, 100)
        
        print("Returning home...")
        controller.go_home()
        
        print("Test complete!")
        return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_movement()
    else:
        test_connection()
