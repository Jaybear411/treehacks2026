"""
Configuration for the Magic Table - Voice-Controlled Object Retrieval System

Hardware Setup:
- Arduino Uno + CNC Shield V3
- DRV8825 drivers (X, Y, Z used for XY gantry + optional A-slot aux motor)
- 24V power supply for motors
- 4 NEMA 17 stepper motors (1x X-axis, 2x Y-axis, 1x magnet engage/disengage)
- Gantry: 400mm x 400mm (2020 aluminum extrusion)

Dual-Y Motor Wiring (software mirror, no hardware jumper):
  X driver slot → X-axis motor
  Y driver slot → Y-axis motor #1
  Z driver slot → Y-axis motor #2 (mirrored from Y via software)
  A driver slot → magnet lifter motor (engage/disengage rotation)
"""

import json
import os
from dataclasses import dataclass
from typing import Tuple


def _load_axis_lengths():
    """Load measured axis lengths from measure_axis.py output (axis_lengths.json)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axis_lengths.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


@dataclass
class GantryConfig:
    """Physical gantry dimensions and limits."""
    # Gantry dimensions in mm (400mm 2020 extrusions)
    width_mm: float = 400.0   # X axis travel
    height_mm: float = 400.0  # Y axis travel
    
    # Safe margins from edges (mm)
    margin_mm: float = 10.0
    
    # Pickup zone location (where objects are delivered)
    # Y=0 = all the way forward (until Y limit switch); X kept in from left edge.
    pickup_x_mm: float = 20.0
    pickup_y_mm: float = 0.0
    
    # Home position (where magnet rests when idle)
    home_x_mm: float = 0.0
    home_y_mm: float = 0.0
    
    # Cleanup zone — where objects are pushed during cleanup mode
    # Default: right edge of the gantry, preserving each object's Y
    cleanup_x_mm: float = 380.0
    
    # Pixel-space threshold for deciding if an object needs moving.
    # Objects whose center_x is >= this fraction of frame width are
    # considered "already at the clean side" and will be skipped.
    cleanup_threshold_fraction: float = 0.75
    
    # ── Perspective depth correction ──────────────────────────────────
    # Objects closer to the camera (bottom of the frame) appear
    # disproportionately large due to perspective foreshortening.  The
    # simple linear pixel→mm mapping therefore UNDERESTIMATES their
    # true physical distance from the gantry center — the robot
    # undershoots.
    #
    # This correction scales the displacement of the mapped point from
    # the gantry center by a factor that grows with proximity to the
    # camera:
    #
    #   depth_scale = 1 + perspective_correction × t^perspective_exponent
    #
    # where t ∈ [0, 1] is the normalised closeness to the camera
    # (0 = top of frame / far from camera, 1 = bottom / near camera).
    #
    # The result is always clamped to the gantry boundaries so the
    # robot can never exceed its physical travel limits.
    #
    # Tuning guide:
    #   perspective_correction = 0.0   → disabled (pure linear mapping)
    #   perspective_correction = 0.10  → mild (camera nearly overhead)
    #   perspective_correction = 0.15  → moderate (slight camera tilt)
    #   perspective_correction = 0.25  → aggressive (noticeable tilt)
    #
    #   perspective_exponent   = 1.0   → linear ramp (uniform boost)
    #   perspective_exponent   = 1.5   → gentle curve (default)
    #   perspective_exponent   = 2.0   → quadratic (minimal near center,
    #                                     aggressive near camera)
    perspective_correction: float = 0.28
    perspective_exponent: float = 1.5
    
    # Global “move more” factor: scale displacement from gantry center
    # so the robot moves further toward the target everywhere, then clamp.
    # 1.0 = no extra reach; 1.08 = 8% further; 1.12 = 12% further.
    # Keeps movement within boundaries via clamping.
    reach_scale: float = 1.10
    
    # Scale for the wide (X) direction only.  Use < 1.0 if the robot
    # goes a little too far in the width direction (e.g. 0.92 = 8% less).
    width_scale: float = 0.92
    
    # Virtual coordinate space used by calibration (marker positions).
    # The homography maps camera pixels to this 0-N range.  Motor commands
    # need actual mm, so the pipeline applies:
    #   motor_mm = calibration_value * (travel_mm / calibration_space)
    calibration_space: float = 400.0
    
    @property
    def min_x(self) -> float:
        return self.margin_mm
    
    @property
    def max_x(self) -> float:
        return self.width_mm - self.margin_mm
    
    @property
    def min_y(self) -> float:
        return self.margin_mm
    
    @property
    def max_y(self) -> float:
        return self.height_mm - self.margin_mm


@dataclass
class GRBLConfig:
    """GRBL/Arduino serial communication settings."""
    # Serial port - update this for your system
    # macOS: /dev/tty.usbmodem* or /dev/tty.usbserial*
    # Linux: /dev/ttyUSB0 or /dev/ttyACM0
    # Windows: COM3, COM4, etc.
    port: str = "/dev/cu.usbmodem11101"
    
    # Baud rate (GRBL default is 115200)
    baud_rate: int = 115200
    
    # Movement speeds (mm/min)
    feed_rate: float = 3000.0       # Normal movement speed
    rapid_rate: float = 5000.0      # Fast positioning speed
    drag_rate: float = 1500.0       # Slower speed when dragging objects
    
    # Acceleration (mm/sec^2) - configured in GRBL $120, $121, $122
    # Updated to 200 to match goto.py / home.py tested values.
    acceleration: float = 200.0
    
    # Steps per mm - depends on your motor/pulley setup
    # DRV8825 jumpers: M0=off, M1=on, M2=on → 1/32 microstepping
    # For GT2 belt (2mm pitch) + 20T pulley + 1/32 microstepping + 200 step motor:
    # steps_per_mm = (200 * 32) / (20 * 2) = 160
    steps_per_mm: float = 160.0
    
    # Dual-Y axis: mirror Y movements to the Z-axis driver slot.
    # Set True when the second Y motor is plugged into the Z driver slot
    # on the CNC Shield V3 (no hardware jumper required).
    mirror_y_to_z: bool = True
    
    # If your two Y-axis motors face opposite directions on the gantry,
    # you need to invert the Z direction in GRBL ($3 direction invert mask).
    # Set $3=4 to invert Z only, $3=6 to invert both Y and Z, etc.
    # This is handled in GRBL firmware settings, not here.
    
    # Z-axis scaling for dual-Y.  The second Y motor (on Z driver) may
    # need slightly less travel to stay synchronized.  Both goto.py and
    # home.py use 0.90 (Z gets 90 % of Y).  Set to 1.0 for 1:1 mirror.
    y_z_scale: float = 0.90
    
    # Coordinate convention.  True when home switches sit at the positive
    # GRBL end and the work area extends in the negative direction.
    # goto.py proved this matches our CNC Shield V3 + DRV8825 wiring.
    negate_grbl_coords: bool = True
    
    # Max rate per axis (mm/min) written to $110 / $111 / $112 on connect.
    # Matches goto.py / home.py proven values.
    max_rate: float = 6000.0
    
    # Hard limits ($21).  False = disabled during normal operation for
    # more reliable movement (no ALARM:1 interruptions).  Homing still
    # uses switch detection via pin polling.  Set True to re-enable.
    enable_hard_limits: bool = False

    # Magnet lifter on the auxiliary axis (e.g., A slot).
    # The controller uses RELATIVE moves for this axis:
    #   - engage_delta_units: rotate to lift/engage magnet
    #   - disengage_delta_units: rotate opposite direction to lower/disengage
    # NOTE: Axis support depends on your GRBL build/wiring. If needed, set
    # this to "Z" (or another controllable axis) to match your firmware.
    magnet_lift_enabled: bool = True
    magnet_axis: str = "A"
    magnet_engage_delta_units: float = 18.0
    magnet_disengage_delta_units: float = -18.0
    magnet_feed_rate: float = 500.0
    magnet_settle_sec: float = 0.25

    # Automatic lift sequencing during drag operations.
    # Typical flow: disengage while traveling -> engage at object -> drag -> disengage.
    auto_disengage_before_travel: bool = True
    auto_disengage_after_drag: bool = True
    
    # Timeout for serial operations (seconds)
    timeout: float = 2.0
    
    # Command acknowledgment timeout
    ack_timeout: float = 30.0


@dataclass 
class CameraConfig:
    """Camera and vision settings."""
    # Camera index (0 = USB camera, 1 = phone/continuity, 2 = built-in)
    index: int = 0
    
    # Resolution (if supported by camera)
    width: int = 1280
    height: int = 720
    
    # Frame rate target
    fps: int = 30
    
    # Digital zoom applied before any processing (1.0 = no zoom).
    # Values > 1.0 crop to the center and resize back (e.g. 1.5 = 50% zoom in).
    zoom_factor: float = 1.7


@dataclass
class CalibrationConfig:
    """Camera-to-gantry calibration settings."""
    # Calibration points: pixel (x, y) -> physical (x, y) mm
    # You'll set these during calibration by moving the magnet to
    # known positions and recording the corresponding pixel coordinates
    
    # Four corner calibration points (pixel coordinates)
    # These get filled in during calibration
    pixel_top_left: Tuple[float, float] = (0.0, 0.0)
    pixel_top_right: Tuple[float, float] = (1280.0, 0.0)
    pixel_bottom_left: Tuple[float, float] = (0.0, 720.0)
    pixel_bottom_right: Tuple[float, float] = (1280.0, 720.0)
    
    # Corresponding physical coordinates (mm)
    physical_top_left: Tuple[float, float] = (0.0, 400.0)
    physical_top_right: Tuple[float, float] = (400.0, 400.0)
    physical_bottom_left: Tuple[float, float] = (0.0, 0.0)
    physical_bottom_right: Tuple[float, float] = (400.0, 0.0)


@dataclass
class MarkerConfig:
    """Blue-dot calibration marker detection settings.
    
    Place 4 blue markers (tape dots) at the gantry corners.
    The auto-calibration system detects them via HSV color filtering
    and maps them to the known physical corner positions.
    """
    # HSV range for blue marker detection
    # Default tuned for blue painter's tape / blue dots under typical lighting
    # Hue: 90-130 covers most blues (OpenCV hue range is 0-179)
    hue_low: int = 90
    hue_high: int = 130
    sat_low: int = 80
    sat_high: int = 255
    val_low: int = 60
    val_high: int = 255
    
    # Contour area filter (pixels^2) — reject noise and overly large blobs
    min_area: int = 200
    max_area: int = 50000
    
    # Minimum circularity (0.0 - 1.0). Dots should be roughly circular.
    # Lower this if your markers are irregular shapes.
    min_circularity: float = 0.3
    
    # Gaussian blur kernel size applied before HSV thresholding (must be odd)
    blur_kernel: int = 5
    
    # Number of frames to average for more stable detection
    detection_frames: int = 5
    
    # Physical positions of the 4 markers (mm) — must match your gantry layout.
    # Order: top-left, top-right, bottom-right, bottom-left (when viewed from camera)
    # These are at the EDGES of the gantry with a small margin.
    # NOTE: "top" in camera view = high Y in physical space (camera is overhead)
    marker_positions_mm: Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ] = (
        (10.0, 390.0),   # top-left in camera  → physical (margin, height-margin)
        (390.0, 390.0),  # top-right in camera → physical (width-margin, height-margin)
        (390.0, 10.0),   # bottom-right        → physical (width-margin, margin)
        (10.0, 10.0),    # bottom-left         → physical (margin, margin)
    )


@dataclass
class VoiceConfig:
    """Voice recognition settings."""
    # Wake word (optional - set to None to always listen)
    wake_word: str | None = None
    
    # Timeout for listening (seconds)
    listen_timeout: float = 5.0
    
    # Phrase time limit (seconds)
    phrase_time_limit: float = 10.0
    
    # Energy threshold for speech detection (adjust based on environment)
    # Lower = more sensitive, Higher = less sensitive
    energy_threshold: int = 300


@dataclass
class DetectionConfig:
    """Object detection settings."""
    # Objects to detect
    prompts: list = None
    
    # Confidence threshold (0.0 - 1.0)
    score_threshold: float = 0.25
    
    # Model to use (OWLv2 is more stable than GroundingDINO)
    # Using non-ensemble variant for faster inference
    model_name: str = "google/owlv2-base-patch16"
    
    # Max image dimension for detection (smaller = faster)
    max_detect_dim: int = 640
    
    # Smoothing half-life (seconds) - lower = more responsive
    smoothing_half_life: float = 0.15
    
    def __post_init__(self):
        if self.prompts is None:
            self.prompts = [
                "keys",
                "airpods", 
                "phone",
                "wallet",
                "pill bottle",
                "glasses",
                "remote",
                "pen",
                "cup",
            ]


@dataclass
class OpenAIVisionConfig:
    """OpenAI settings for vision tasks (scan_table_objects)."""
    # Model to use for vision-based tasks (image input required)
    # gpt-4.1-mini is fast, cheap, and has strong vision capabilities
    model_name: str = "gpt-4.1-mini"


@dataclass
class AnthropicConfig:
    """Anthropic (Claude) settings for text-based NLP tasks."""
    # Model to use for object extraction and conversation (text-only)
    model_name: str = "claude-haiku-4-5"


@dataclass
class ElevenLabsConfig:
    """ElevenLabs TTS settings for spoken responses."""
    # Voice ID (default: "JBFqnCBsd6RMkjVDRZzb" = George)
    # Browse voices at https://elevenlabs.io/voice-library
    voice_id: str = "ql9agxema3MD2v2abuqw" #jarvis
    
    # Model: "eleven_flash_v2_5" for low latency, "eleven_multilingual_v2" for quality
    model_id: str = "eleven_flash_v2_5"
    
    # Output format
    output_format: str = "mp3_44100_128"


# Default configurations — gantry size from measured axis lengths when present
_axis = _load_axis_lengths()
_width_mm = _axis.get("x_mm") or 400.0
_height_mm = _axis.get("y_mm") or 400.0
GANTRY = GantryConfig(
    width_mm=_width_mm,
    height_mm=_height_mm,
    cleanup_x_mm=_width_mm - 10.0,  # right edge minus margin
)
GRBL = GRBLConfig()
CAMERA = CameraConfig()
CALIBRATION = CalibrationConfig()
MARKER = MarkerConfig()
VOICE = VoiceConfig()
DETECTION = DetectionConfig()
OPENAI_VISION = OpenAIVisionConfig()
ANTHROPIC = AnthropicConfig()
ELEVENLABS = ElevenLabsConfig()


def print_config():
    """Print current configuration."""
    print("=" * 50)
    print("Magic Table Configuration")
    print("=" * 50)
    print(f"\nGantry: {GANTRY.width_mm}mm x {GANTRY.height_mm}mm")
    print(f"Pickup zone: ({GANTRY.pickup_x_mm}, {GANTRY.pickup_y_mm}) mm")
    print(f"\nGRBL Port: {GRBL.port}")
    print(f"Feed rate: {GRBL.feed_rate} mm/min")
    print(f"Steps/mm: {GRBL.steps_per_mm}")
    print(f"\nCamera index: {CAMERA.index}")
    print(f"Resolution: {CAMERA.width}x{CAMERA.height}")
    print(f"\nDetection prompts: {DETECTION.prompts}")
    print(f"Score threshold: {DETECTION.score_threshold}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
