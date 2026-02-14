"""
Configuration for the Magic Table - Voice-Controlled Object Retrieval System

Hardware Setup:
- Arduino Uno + CNC Shield V3
- DRV8825 drivers (X, Y, A axes - A cloned to Y via jumper)
- 24V power supply for motors
- 3 NEMA 17 stepper motors
- Gantry: 400mm x 400mm (2020 aluminum extrusion)
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class GantryConfig:
    """Physical gantry dimensions and limits."""
    # Gantry dimensions in mm (400mm 2020 extrusions)
    width_mm: float = 400.0   # X axis travel
    height_mm: float = 400.0  # Y axis travel
    
    # Safe margins from edges (mm)
    margin_mm: float = 10.0
    
    # Pickup zone location (where objects are delivered)
    # Default: front-left corner
    pickup_x_mm: float = 20.0
    pickup_y_mm: float = 20.0
    
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
    port: str = "/dev/tty.usbmodem1101"
    
    # Baud rate (GRBL default is 115200)
    baud_rate: int = 115200
    
    # Movement speeds (mm/min)
    feed_rate: float = 3000.0       # Normal movement speed
    rapid_rate: float = 5000.0      # Fast positioning speed
    drag_rate: float = 1500.0       # Slower speed when dragging objects
    
    # Acceleration (mm/sec^2) - configured in GRBL $120, $121
    acceleration: float = 500.0
    
    # Steps per mm - depends on your motor/pulley setup
    # For GT2 belt (2mm pitch) + 20T pulley + 1/32 microstepping + 200 step motor:
    # steps_per_mm = (200 * 32) / (20 * 2) = 160
    steps_per_mm: float = 160.0
    
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
class GeminiConfig:
    """Gemini LLM settings for natural language command parsing."""
    # Model to use for NLP object extraction and conversation
    model_name: str = "gemini-2.0-flash"


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


# Default configurations
GANTRY = GantryConfig()
GRBL = GRBLConfig()
CAMERA = CameraConfig()
CALIBRATION = CalibrationConfig()
MARKER = MarkerConfig()
VOICE = VoiceConfig()
DETECTION = DetectionConfig()
GEMINI = GeminiConfig()
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
