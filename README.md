# Magic Table - Voice-Controlled Object Retrieval

A voice-controlled robotic system for accessibility. Uses computer vision to identify objects on a tabletop and an XY magnet gantry to drag them to a pickup zone.

**Core use case:** Accessibility for blind/mobility-impaired users — "voice-controlled physical retrieval" in a clean, magical demo.

## How It Works

1. **Camera + Vision Model** watches the tabletop, identifies objects (keys, phone, glasses, etc.), and tracks their (x, y) position
2. **XY Gantry** with powerful magnets moves under the table surface
3. **Metal pucks** attached to objects couple with magnets underneath
4. **Voice commands** like "bring me the keys" route the magnet to pull that object to the pickup zone

## Hardware Requirements

### Electronics Stack

| Component | Purpose |
|-----------|---------|
| Arduino Uno | Controller (runs GRBL firmware) |
| CNC Shield V3 | Motor driver interface (plugs onto Uno) |
| DRV8825 Drivers (x3) | Stepper motor drivers (X, Y, A slots) |
| 24V Power Supply | Motor power (NOT to Arduino!) |
| NEMA 17 Steppers (x3) | Two for Y-axis (ganged), one for X-axis |
| Webcam | Overhead view of table surface |

### Mechanical Stack

| Component | Specification |
|-----------|---------------|
| 2020 Aluminum Extrusion | 400mm length (x3) |
| Linear Rails/V-wheels | For smooth gantry movement |
| GT2 Belts + Pulleys | Motion transmission |
| Strong Neodymium Magnets | Under-table magnet carriage |
| Metal Pucks | Attached to objects for magnetic coupling |

### Wiring Diagram

```
                    ┌─────────────────┐
                    │   24V Power     │
                    │   Supply        │
                    └────────┬────────┘
                             │
    ┌────────────────────────┴────────────────────────┐
    │                 CNC Shield V3                    │
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐               │
    │  │  X  │ │  Y  │ │  Z  │ │  A  │  ← Drivers    │
    │  │DRV  │ │DRV  │ │empty│ │DRV  │               │
    │  └──┬──┘ └──┬──┘ └─────┘ └──┬──┘               │
    │     │      │               │                   │
    │  Motor   Motor           Motor                 │
    │   #3      #1              #2                   │
    │  (X)     (Y)            (Y clone)              │
    │                                                │
    │  Set A→Y jumper to clone A axis to Y!          │
    └─────────────────────────────────┬──────────────┘
                                      │ (sits on top)
                    ┌─────────────────┴──────────────┐
                    │        Arduino Uno              │
                    │         (GRBL)                  │
                    └─────────────────┬──────────────┘
                                      │ USB
                    ┌─────────────────┴──────────────┐
                    │         Computer                │
                    │   (runs this Python code)       │
                    └────────────────────────────────┘
```

## Software Setup

### 1. Flash GRBL to Arduino

1. Download [GRBL](https://github.com/grbl/grbl)
2. Flash to Arduino Uno via Arduino IDE
3. Configure GRBL settings (see below)

### 2. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies

```

**Note for macOS:** PyAudio may require portaudio:
```bash
brew install portaudio
pip install pyaudio
```

### 3. Configure Serial Port

Edit `config.py` and set the correct serial port:

```python
# macOS
port: str = "/dev/tty.usbmodem1101"

# Linux
port: str = "/dev/ttyUSB0"

# Windows
port: str = "COM3"
```

Find your port:
```bash
# macOS/Linux
ls /dev/tty.*

# Or use Python
python -c "from grbl_controller import GRBLController; print(GRBLController.list_ports())"
```

### 4. GRBL Configuration

Connect via serial terminal (115200 baud) and configure:

```
$100=160    # X steps/mm (adjust for your setup)
$101=160    # Y steps/mm
$110=5000   # X max rate mm/min
$111=5000   # Y max rate mm/min
$120=500    # X acceleration mm/sec^2
$121=500    # Y acceleration
$130=400    # X max travel mm
$131=400    # Y max travel mm
```

## Usage

### Run Full System

```bash
python main.py
```

### Test Individual Components

```bash
# Test vision only (no motor/voice)
python main.py --test-vision

# Test voice recognition
python main.py --test-voice

# Test motor control
python main.py --test-motor

# Run without voice
python main.py --no-voice

# Run without motor (vision demo)
python main.py --no-motor
```

### Calibrate Camera-to-Gantry Mapping

```bash
python main.py --calibrate
```

This walks you through clicking four corners to map camera pixels to physical coordinates.

## Voice Commands

| Command | Action |
|---------|--------|
| "bring me the keys" | Fetch keys to pickup zone |
| "get my phone" | Fetch phone to pickup zone |
| "where is the wallet" | Highlight wallet location |
| "stop" | Emergency stop |
| "go home" | Return magnet to home position |

You can also just say the object name: "keys", "phone", "glasses"

## Detectable Objects

Default objects (configurable in `config.py`):
- keys
- airpods
- phone
- wallet
- pill bottle
- glasses
- remote
- pen
- cup

## File Structure

```
treehacks2026/
├── main.py              # Main orchestration
├── config.py            # Configuration settings
├── grbl_controller.py   # Arduino/GRBL serial control
├── object_tracker.py    # Vision-based object detection
├── voice_control.py     # Speech recognition
├── calibration.py       # Camera-to-gantry calibration
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Troubleshooting

### Camera not found
- Check camera index in `config.py` (try 0, 1, 2)
- Ensure no other app is using the camera

### Motor not responding
- Check USB connection
- Verify serial port in `config.py`
- Ensure GRBL is flashed correctly
- Check 24V power to CNC shield

### Voice not recognized
- Ensure microphone is working
- Calibrate ambient noise (automatic on startup)
- Speak clearly and at normal volume
- Check internet connection (uses Google Speech API)

### Object detection poor
- Improve lighting (even, diffused)
- Adjust `score_threshold` in config (lower = more sensitive)
- Try different objects/backgrounds
- Camera should be overhead with clear view

## Development

### Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Voice     │     │   Vision     │     │    Motor     │
│  Controller  │     │   Tracker    │     │  Controller  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │    ┌───────────────┴───────────────┐    │
       └────►       MagicTable (main.py)    ◄────┘
             │  - Orchestrates components   │
             │  - Coordinate transforms     │
             │  - Command processing        │
             └──────────────────────────────┘
```

### Adding New Objects

Edit `config.py`:
```python
@dataclass
class DetectionConfig:
    prompts: list = None
    
    def __post_init__(self):
        if self.prompts is None:
            self.prompts = [
                "keys",
                "phone",
                "your_new_object",  # Add here
            ]
```

## License

MIT License - TreeHacks 2026

## Credits

- Zero-shot object detection: [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- Motor control: [GRBL](https://github.com/grbl/grbl)
- Speech recognition: [SpeechRecognition](https://github.com/Uberi/speech_recognition)
