"""
Magic Table - Voice-Controlled Object Retrieval System

A voice-controlled robotic system that uses computer vision to identify
objects on a tabletop and moves them to a pickup zone using an XY gantry
with magnets.

Core Use Case: Accessibility for blind/mobility-impaired users.

Hardware Stack:
- Arduino Uno + CNC Shield V3 running GRBL
- DRV8825 stepper drivers (X, Y, Z slots — Z mirrors Y in software for dual-Y gantry)
- 24V power supply
- 3x NEMA 17 stepper motors (1x X-axis, 2x Y-axis)
- 400mm x 400mm gantry (2020 aluminum extrusion)
- Camera (overhead view of table)
- Magnets under table surface + metal pucks on objects

Usage:
    python main.py              # Run full system
    python main.py --no-voice   # Run without voice control
    python main.py --no-motor   # Run without motor control (vision only)
    python main.py --calibrate  # Run calibration wizard
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Optional, Tuple

import cv2
try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None

# Audio file paths (relative to this script)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_YES_RIGHT_AWAY = os.path.join(_SCRIPT_DIR, "yesrightaway.mp3")
AUDIO_SHUTTING_DOWN = os.path.join(_SCRIPT_DIR, "shuttingdown.mp3")
AUDIO_HAPPY_TO_HELP = os.path.join(_SCRIPT_DIR, "happytohelp.mp3")


def play_audio(filepath: str, block: bool = False):
    """
    Play an mp3 file using macOS afplay.

    Args:
        filepath: Path to the audio file.
        block: If True, wait for playback to finish before returning.
    """
    if not os.path.exists(filepath):
        print(f"Audio file not found: {filepath}")
        return
    try:
        proc = subprocess.Popen(
            ["afplay", filepath],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if block:
            proc.wait()
    except Exception as e:
        print(f"Audio playback error: {e}")

from config import GANTRY, GRBL, CAMERA, DETECTION, print_config
from grbl_controller import GRBLController
from object_tracker import ObjectTracker, TrackedObject
from voice_control import VoiceController, VoiceCommand, CommandType
from nlp_voice import NLPVoiceController
# Calibration is no longer used — pixel→mm conversion is simple flip+scale.
# from calibration import CoordinateTransformer, CalibrationWizard, AutoCalibrator


class MagicTable:
    """
    Main controller that orchestrates vision, voice, and motor control.
    
    This is the central coordinator that:
    1. Tracks objects via computer vision
    2. Listens for voice commands
    3. Translates pixel coordinates to physical coordinates
    4. Controls the gantry to retrieve objects
    """
    
    def __init__(self, 
                 enable_voice: bool = True,
                 enable_motor: bool = True,
                 target_label: str = None):
        """
        Initialize the Magic Table system.
        
        Args:
            enable_voice: Enable voice recognition (NLP push-to-talk)
            enable_motor: Enable motor control
            target_label: Fixed label for 'V' key detection (optional)
        """
        self.enable_voice = enable_voice
        self.enable_motor = enable_motor
        self.target_label = target_label
        
        # Components
        self.tracker: Optional[ObjectTracker] = None
        self.nlp_voice: Optional[NLPVoiceController] = None
        self.voice: Optional[VoiceController] = None
        self.motor: Optional[GRBLController] = None
        self.transformer = None  # No longer used (flip+scale instead)
        
        # State
        self.running = False
        self._busy = False  # Currently executing a command
        self._last_command: Optional[VoiceCommand] = None
        
        # Last detected object coordinates (pixel)
        self._last_result: Optional[Tuple[str, float, float]] = None
        
        # Status message for display
        self._status_message = "Initializing..."
        
        # Hold-to-talk keyboard state (global listener)
        self._keyboard_listener = None
        self._keys_down: set[str] = set()
        self._space_hold_latched = False
        self._c_hold_latched = False
    
    def _start_key_listener(self):
        """Start global key listener for hold-to-talk."""
        if pynput_keyboard is None or self._keyboard_listener is not None:
            return
        
        def on_press(key):
            if key == pynput_keyboard.Key.space:
                self._keys_down.add("space")
                return
            try:
                ch = key.char.lower() if key.char else None
                if ch == "c":
                    self._keys_down.add("c")
            except Exception:
                pass
        
        def on_release(key):
            if key == pynput_keyboard.Key.space:
                self._keys_down.discard("space")
                return
            try:
                ch = key.char.lower() if key.char else None
                if ch == "c":
                    self._keys_down.discard("c")
            except Exception:
                pass
        
        self._keyboard_listener = pynput_keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        self._keyboard_listener.daemon = True
        self._keyboard_listener.start()
    
    def _stop_key_listener(self):
        """Stop global key listener."""
        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        self._keys_down.clear()
    
    def _is_key_held(self, key_name: str) -> bool:
        return key_name in self._keys_down
        
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns True if successful.
        """
        print("\n" + "="*60)
        print("MAGIC TABLE - Voice-Controlled Object Retrieval")
        print("="*60 + "\n")
        
        # Print configuration
        print_config()
        print()
        
        # Initialize object tracker FIRST (opens camera + loads model)
        # Camera is opened before model load so macOS locks the correct
        # device index immediately (avoids Continuity Camera stealing it).
        print("\nInitializing object tracker...")
        self.tracker = ObjectTracker(prompts=DETECTION.prompts)
        if not self.tracker.start():
            print("ERROR: Failed to start object tracker")
            return False
        
        # Coordinate conversion is now simple flip + scale (no homography needed).
        # See _pixel_to_physical() — pixel coords are flipped on Y and scaled
        # to gantry mm using CAMERA and GANTRY dimensions directly.
        self.transformer = None
        print(f"\nCoordinate mapping: flip Y + scale")
        print(f"  Camera {CAMERA.width}x{CAMERA.height} px → "
              f"Gantry {GANTRY.width_mm:.1f}x{GANTRY.height_mm:.1f} mm")
        
        # Initialize motor controller
        if self.enable_motor:
            print("\nInitializing motor controller...")
            self.motor = GRBLController()
            if not self.motor.connect():
                print("WARNING: Motor control unavailable")
                self.motor = None
            else:
                # Set current position as home (soft home)
                self.motor.soft_home()
        
        # Initialize NLP voice controller (push-to-talk + Claude/OpenAI)
        if self.enable_voice:
            print("\nInitializing NLP voice controller...")
            try:
                self.nlp_voice = NLPVoiceController()
                self.nlp_voice.calibrate()
                if pynput_keyboard is None:
                    print("WARNING: pynput not installed; hold-to-talk disabled (tap mode fallback).")
                else:
                    self._start_key_listener()
            except Exception as e:
                print(f"WARNING: NLP voice controller failed to initialize: {e}")
                print("Voice control will be unavailable. Check your ANTHROPIC_API_KEY.")
                self.nlp_voice = None
        
        self._status_message = "Ready - Press SPACE to speak a command"
        print("\n" + "="*60)
        print("SYSTEM READY")
        print("="*60)
        print("\nKeyboard controls:")
        print("  HOLD SPACE - Push-to-talk: say what object to find")
        print("          e.g. 'Hey Jarvis, get me the pill bottle'")
        print("          or   'Clean up this table'")
        print("          or   'What's on the table?'")
        print("  HOLD C     - Conversation mode: chat with Jarvis")
        print("  D     - Describe table: tell me what's on the table")
        print("  L     - Cleanup mode: move all objects to one side")
        if self.target_label:
            print(f"  V     - Run VLM detection for '{self.target_label}'")
        print("  H     - Return magnet to home")
        print("  S     - Emergency stop")
        print("  Q     - Quit")
        print("="*60 + "\n")
        
        return True
    
    def _on_voice_command(self, command: VoiceCommand):
        """Callback when a voice command is recognized."""
        self._last_command = command
        print(f"\n>>> Voice command: {command.command_type.value} - {command.object_name or 'N/A'}")
    
    def _pixel_to_physical(self, px: float, py: float) -> tuple:
        """Convert pixel coordinates to motor-ready physical mm.

        Steps:
          1. Flip X and Y (camera left→gantry right, camera top→gantry far)
          2. Scale pixel range to gantry travel (mm)
          3. Apply perspective depth correction (near-camera objects pushed further)
          4. Apply global reach_scale so the robot moves further toward every target
          5. Clamp to physical boundaries so we never exceed travel limits
        """
        # ── Step 1-2: Linear mapping (with horizontal + vertical flip) ─
        # Flip X: camera left (px=0) → gantry right (max X),
        #         camera right (px=width) → gantry left (0)
        x = (CAMERA.width - px) * (GANTRY.width_mm / CAMERA.width)
        # Flip Y: camera top (py=0) → far side of gantry (max Y),
        #         camera bottom (py=height) → front of gantry (Y=0)
        y = (CAMERA.height - py) * (GANTRY.height_mm / CAMERA.height)

        # ── Step 3: Perspective depth correction ──────────────────────
        # Objects closer to the camera (bottom of frame, high py) appear
        # larger due to perspective.  The linear mapping underestimates
        # their true physical distance from the gantry center.  We scale
        # the displacement from center by a factor that grows with
        # proximity to the camera edge.
        #
        #   t ∈ [0, 1]: normalised closeness to camera
        #       0 = top of frame (far from camera)
        #       1 = bottom of frame (near camera)
        #
        #   depth_scale = 1 + correction × t^exponent
        #
        # See config.py GantryConfig for tuning guidance.
        gantry_cx = GANTRY.width_mm / 2.0
        gantry_cy = GANTRY.height_mm / 2.0

        if GANTRY.perspective_correction > 0:
            t = py / CAMERA.height  # 0 at top, 1 at bottom (near camera)
            depth_scale = 1.0 + GANTRY.perspective_correction * (
                t ** GANTRY.perspective_exponent
            )
            x = gantry_cx + (x - gantry_cx) * depth_scale
            y = gantry_cy + (y - gantry_cy) * depth_scale

        # Global “move more”: scale displacement from center so the robot
        # travels further toward the target everywhere (still clamped below).
        if GANTRY.reach_scale != 1.0:
            x = gantry_cx + (x - gantry_cx) * GANTRY.reach_scale
            y = gantry_cy + (y - gantry_cy) * GANTRY.reach_scale

        # Scale down the wide (X) direction if it tends to go too far.
        if GANTRY.width_scale != 1.0:
            x = gantry_cx + (x - gantry_cx) * GANTRY.width_scale

        # ── Step 4: Clamp to physical travel limits ───────────────────
        x = max(0.0, min(GANTRY.width_mm, x))
        y = max(0.0, min(GANTRY.height_mm, y))
        return x, y
    
    def fetch_object(self, object_name: str) -> bool:
        """
        Fetch an object and bring it to the pickup zone.
        
        Args:
            object_name: Name of the object to fetch
            
        Returns:
            True if successful
        """
        if self._busy:
            print("System busy - please wait")
            return False
        
        self._busy = True
        self._status_message = f"Fetching {object_name}..."
        
        try:
            # Find the object
            obj = self.tracker.find_object(object_name)
            if obj is None:
                print(f"Object not found: {object_name}")
                print(f"Visible objects: {self.tracker.list_visible_objects()}")
                self._status_message = f"Can't find {object_name}"
                return False
            
            # Get pixel position
            px, py = obj.center_x, obj.center_y
            print(f"Found {object_name} at pixel ({px:.1f}, {py:.1f})")
            
            # Convert to physical coordinates
            phys_x, phys_y = self._pixel_to_physical(px, py)
            print(f"Physical position: ({phys_x:.1f}, {phys_y:.1f}) mm")
            
            # Move the gantry
            if self.motor and self.motor.connected:
                # Move to object and drag to pickup zone
                success = self.motor.drag_object_to(
                    from_x=phys_x,
                    from_y=phys_y,
                    to_x=GANTRY.pickup_x_mm,
                    to_y=GANTRY.pickup_y_mm
                )
                
                if success:
                    self._status_message = f"Delivered {object_name} to pickup zone"
                    print(f"Successfully delivered {object_name}")
                    
                    # Speak confirmation (text-to-speech could be added here)
                    # For now, just print
                    print(f"*** {object_name.upper()} IS READY FOR PICKUP ***")
                else:
                    self._status_message = f"Failed to move {object_name}"
                    return False
            else:
                print("Motor control not available - simulation mode")
                self._status_message = f"[SIM] Would fetch {object_name}"
                time.sleep(1)  # Simulate movement
            
            return True
            
        finally:
            self._busy = False
    
    def locate_object(self, object_name: str) -> Optional[TrackedObject]:
        """
        Find and highlight an object without moving it.
        
        Args:
            object_name: Name of the object to locate
            
        Returns:
            TrackedObject if found, None otherwise
        """
        obj = self.tracker.find_object(object_name)
        if obj:
            print(f"Found {object_name} at ({obj.center_x:.1f}, {obj.center_y:.1f}) pixels")
            self._status_message = f"Found {object_name}"
        else:
            print(f"Object not found: {object_name}")
            self._status_message = f"Can't find {object_name}"
        return obj
    
    def go_home(self):
        """Return the magnet to home position."""
        self._status_message = "Returning home..."
        if self.motor and self.motor.connected:
            self.motor.go_home()
        self._status_message = "Ready"
    
    def emergency_stop(self):
        """Emergency stop all motion."""
        print("!!! EMERGENCY STOP !!!")
        self._status_message = "STOPPED"
        if self.motor:
            self.motor.stop()
    
    def process_command(self, command: VoiceCommand):
        """Process a voice command."""
        if command.command_type == CommandType.FETCH:
            if command.object_name:
                self.fetch_object(command.object_name)
                
        elif command.command_type == CommandType.LOCATE:
            if command.object_name:
                self.locate_object(command.object_name)
                
        elif command.command_type == CommandType.STOP:
            self.emergency_stop()
            
        elif command.command_type == CommandType.HOME:
            self.go_home()
    
    def voice_detect_pipeline(self, hold_check=None):
        """
        Full voice-driven pipeline:
        1. Record audio (push-to-talk)
        2. Transcribe speech
        3. Extract object name via Claude (or detect goodbye)
        4. Run VLM detection for that object
        5. Return and display coordinates
        """
        if not self.nlp_voice:
            self._status_message = "Voice not available (check ANTHROPIC_API_KEY)"
            return

        self._status_message = "LISTENING... speak now"
        # Force a display update so the user sees the status
        self._force_display_update()

        # Step 1-3: Listen and extract object name via Claude
        result = self.nlp_voice.listen_and_extract(hold_check=hold_check)

        if not result:
            self._status_message = "Didn't catch that. Press SPACE to try again."
            return

        # Check if the user said goodbye
        if result == NLPVoiceController.GOODBYE:
            print("User said goodbye.")
            self._status_message = "Goodbye! Happy to help."
            play_audio(AUDIO_HAPPY_TO_HELP)
            self._force_display_update()
            return

        # Check if the user requested cleanup
        if result == NLPVoiceController.CLEANUP:
            print("User requested cleanup.")
            play_audio(AUDIO_YES_RIGHT_AWAY)
            self.cleanup_pipeline()
            return

        # Check if the user asked what's on the table
        if result == NLPVoiceController.DESCRIBE_TABLE:
            print("User asked what's on the table.")
            self.describe_table_pipeline()
            return

        object_name = result

        # User asked for an object — acknowledge with "Yes, right away"
        play_audio(AUDIO_YES_RIGHT_AWAY)

        # Step 4: Run VLM detection for the extracted object
        self._status_message = f"Scanning for '{object_name}'..."
        self._force_display_update()

        detected = self.tracker.run_detection(target_label=object_name)

        if detected:
            obj = detected[0]
            px, py = obj.center_x, obj.center_y

            # Compute physical coordinates
            phys_x, phys_y = self._pixel_to_physical(px, py)

            self._last_result = (object_name, px, py)

            print("\n" + "=" * 60)
            print(f"RESULT: '{object_name}'")
            print(f"  Pixel coords:    ({px:.1f}, {py:.1f})")
            print(f"  Physical coords: ({phys_x:.1f}, {phys_y:.1f}) mm")
            print(f"  Confidence:      {obj.confidence:.2f}")
            print("=" * 60 + "\n")

            # ── Move the gantry: home → object → front ──
            if self.motor and self.motor.connected:
                self._status_message = f"Fetching '{object_name}'..."
                self._force_display_update()
                success = self.motor.drag_object_to(
                    from_x=phys_x,
                    from_y=phys_y,
                    to_x=GANTRY.pickup_x_mm,
                    to_y=GANTRY.pickup_y_mm,
                )
                if success:
                    self._status_message = f"Delivered '{object_name}' to pickup zone!"
                    print(f"*** {object_name.upper()} IS READY FOR PICKUP ***")
                else:
                    self._status_message = f"Motor move failed for '{object_name}'"
            else:
                self._status_message = (
                    f"Found '{object_name}' at ({phys_x:.0f}, {phys_y:.0f}) mm "
                    f"[motor not connected]"
                )
        else:
            self._last_result = None
            self._status_message = f"'{object_name}' not found on table. Press SPACE to try again."

    def describe_table_pipeline(self):
        """
        Describe what's on the table — scan with OpenAI Vision and report back.

        When triggered via voice the response is spoken aloud through
        ElevenLabs TTS.  The result is also shown on the OpenCV display.
        """
        self._status_message = "Looking at the table..."
        self._force_display_update()

        # Grab the latest frame
        frame = self.tracker.get_frame()
        if frame is None:
            frame = self.tracker.capture_frame()
        if frame is None:
            self._status_message = "Can't see the table (no camera frame)"
            return

        if not self.nlp_voice:
            self._status_message = "Vision not available (check OPENAI_API_KEY for vision)"
            return

        objects = self.nlp_voice.scan_table_objects(frame)
        sentence = NLPVoiceController.objects_to_sentence(objects)

        print(f"\nDESCRIBE TABLE: {sentence}")
        self._status_message = sentence
        self._force_display_update()

        # Speak the answer aloud
        self.nlp_voice.speak(sentence)

    def cleanup_pipeline(self):
        """
        Cleanup mode — consolidate all scattered objects to one side of the table.

        Steps:
        1. Capture scene, use OpenAI Vision to list every object on the table.
        2. For each object, run VLM detection to get pixel coordinates.
        3. Convert to physical coords; skip if already on the "clean" side.
        4. Drag the object to the cleanup edge (preserving its Y position).
        5. Return home when finished.
        """
        if self._busy:
            print("System busy — please wait")
            return

        self._busy = True
        self._status_message = "CLEANUP: Scanning table..."
        self._force_display_update()

        try:
            # ── Step 1: Capture scene & identify objects via OpenAI Vision ──
            frame = self.tracker.get_frame()
            if frame is None:
                frame = self.tracker.capture_frame()
            if frame is None:
                self._status_message = "Cleanup failed: no camera frame"
                return

            if not self.nlp_voice:
                self._status_message = "Cleanup requires OpenAI Vision (check OPENAI_API_KEY)"
                return

            objects = self.nlp_voice.scan_table_objects(frame)

            if not objects:
                self._status_message = "Cleanup: no objects found on table"
                return

            print(f"\nCLEANUP: Found {len(objects)} objects: {objects}")
            self._status_message = f"CLEANUP: Found {len(objects)} objects — moving..."
            self._force_display_update()

            # ── Steps 2-4: Process each object sequentially ────────────
            moved_count = 0
            for i, obj_name in enumerate(objects):
                tag = f"[{i + 1}/{len(objects)}]"
                self._status_message = f"CLEANUP {tag}: Detecting '{obj_name}'..."
                self._force_display_update()

                # Re-capture a fresh frame before each detection so the
                # tracker has an up-to-date view (previous object may have
                # moved, changing the scene).
                self.tracker.capture_frame()

                detected = self.tracker.run_detection(target_label=obj_name)
                if not detected:
                    print(f"  {tag} Could not find '{obj_name}' — skipping")
                    continue

                obj = detected[0]
                px, py = obj.center_x, obj.center_y

                # Check if the object is already on the clean side
                frame_width = CAMERA.width
                threshold_px = frame_width * GANTRY.cleanup_threshold_fraction
                if px >= threshold_px:
                    print(f"  {tag} '{obj_name}' already at cleanup edge — skipping")
                    continue

                # Convert to physical coordinates
                phys_x, phys_y = self._pixel_to_physical(px, py)

                # Destination: cleanup edge, preserve Y
                dest_x = GANTRY.cleanup_x_mm
                dest_y = phys_y

                self._status_message = (
                    f"CLEANUP {tag}: Moving '{obj_name}' → ({dest_x:.0f}, {dest_y:.0f})"
                )
                self._force_display_update()

                print(
                    f"  {tag} Moving '{obj_name}' from "
                    f"({phys_x:.0f}, {phys_y:.0f}) to ({dest_x:.0f}, {dest_y:.0f})"
                )

                if self.motor and self.motor.connected:
                    success = self.motor.drag_object_to(
                        from_x=phys_x,
                        from_y=phys_y,
                        to_x=dest_x,
                        to_y=dest_y,
                    )
                    if success:
                        moved_count += 1
                    else:
                        print(f"  {tag} Motor failed for '{obj_name}'")
                else:
                    # Simulation mode
                    print(f"  {tag} [SIM] Would move '{obj_name}'")
                    moved_count += 1
                    time.sleep(0.5)

            # ── Step 5: Return home ────────────────────────────────────
            if self.motor and self.motor.connected:
                self.motor.go_home()

            self._status_message = (
                f"CLEANUP DONE: Moved {moved_count}/{len(objects)} objects to edge"
            )
            print(f"\nCLEANUP COMPLETE — moved {moved_count}/{len(objects)} objects\n")

        finally:
            self._busy = False

    def conversation_pipeline(self, hold_check=None):
        """
        Conversational mode:
        1. Record audio (push-to-talk)
        2. Transcribe speech
        3. Send to Claude for conversational response
        4. Speak the response via ElevenLabs TTS
        """
        if not self.nlp_voice:
            self._status_message = "Voice not available (check ANTHROPIC_API_KEY)"
            return

        self._status_message = "LISTENING... (conversation mode)"
        self._force_display_update()

        # Record and transcribe
        text = self.nlp_voice.record_and_transcribe(hold_check=hold_check)

        if not text:
            self._status_message = "Didn't catch that. Press C to try again."
            return

        self._status_message = f"You said: '{text}' — thinking..."
        self._force_display_update()

        # Get Claude's conversational response
        reply = self.nlp_voice.converse(text)

        if not reply:
            self._status_message = "No response from Claude."
            return

        self._status_message = f"Jarvis: {reply}"
        self._force_display_update()

        # Speak the response via ElevenLabs
        self.nlp_voice.speak(reply)

        self._status_message = f"Jarvis: {reply}"

    def _force_display_update(self):
        """Redraw the display immediately (used during blocking operations)."""
        frame = self.tracker.get_frame()
        if frame is not None:
            display = self.tracker.draw_detections(frame)
            h, w = display.shape[:2]
            cv2.rectangle(display, (0, h - 40), (w, h), (0, 0, 0), -1)
            cv2.putText(display, self._status_message, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow("Magic Table", display)
            cv2.waitKey(1)

    def run(self):
        """
        Main run loop.
        
        This is the main event loop that:
        1. Captures camera frames
        2. Waits for user input
        3. SPACE: push-to-talk → Claude NLP → VLM → coordinates
        4. V: manual VLM detection with preset label
        """
        self.running = True
        
        try:
            while self.running:
                # Capture frame (NO detection - that's on-demand)
                frame = self.tracker.capture_frame()
                if frame is None:
                    continue
                
                # Draw display with any existing detections
                display = self.tracker.draw_detections(frame)
                
                # Draw instructions at top
                h, w = display.shape[:2]
                cv2.putText(display, "SPACE: fetch | C: chat | D: describe | L: cleanup | Q: quit",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                
                # Draw object list if any detected
                objects = self.tracker.list_visible_objects()
                if objects:
                    obj_text = "Detected: " + ", ".join(objects)
                    cv2.putText(display, obj_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show last result coordinates
                if self._last_result:
                    name, rx, ry = self._last_result
                    result_text = f"{name}: ({rx:.0f}, {ry:.0f})px"
                    cv2.putText(display, result_text, (w - 300, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show motor status
                if self.motor and self.motor.connected:
                    pos = self.motor.position
                    motor_text = f"Magnet: ({pos[0]:.0f}, {pos[1]:.0f}) mm"
                    cv2.putText(display, motor_text, (w - 200, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Draw status bar at bottom
                cv2.rectangle(display, (0, h - 40), (w, h), (0, 0, 0), -1)
                cv2.putText(display, self._status_message, (10, h - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Display
                cv2.imshow("Magic Table", display)
                
                # Handle hold-to-talk (SPACE/C). Triggers once per hold.
                if self.enable_voice and self.nlp_voice and self._keyboard_listener:
                    if self._is_key_held("space") and not self._space_hold_latched:
                        self._space_hold_latched = True
                        self.voice_detect_pipeline(
                            hold_check=lambda: self._is_key_held("space")
                        )
                    elif not self._is_key_held("space"):
                        self._space_hold_latched = False
                    
                    if self._is_key_held("c") and not self._c_hold_latched:
                        self._c_hold_latched = True
                        self.conversation_pipeline(
                            hold_check=lambda: self._is_key_held("c")
                        )
                    elif not self._is_key_held("c"):
                        self._c_hold_latched = False
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord(' ') and not self._keyboard_listener:
                    # SPACE: Push-to-talk voice pipeline
                    self.voice_detect_pipeline()
                elif key == ord('v') and self.target_label:
                    # V: Manual VLM detection with preset label
                    self._status_message = f"Detecting '{self.target_label}'..."
                    self._force_display_update()
                    detected = self.tracker.run_detection(target_label=self.target_label)
                    if detected:
                        obj = detected[0]
                        self._status_message = (
                            f"Found '{self.target_label}' at "
                            f"({obj.center_x:.0f}, {obj.center_y:.0f})"
                        )
                    else:
                        self._status_message = f"'{self.target_label}' not found."
                elif key == ord('c') and not self._keyboard_listener:
                    # C: Conversation mode
                    self.conversation_pipeline()
                elif key == ord('d'):
                    # D: Describe table — what's on the table?
                    self.describe_table_pipeline()
                elif key == ord('l'):
                    # L: Cleanup mode — move all objects to one side
                    play_audio(AUDIO_YES_RIGHT_AWAY)
                    self.cleanup_pipeline()
                elif key == ord('h'):
                    self.go_home()
                elif key == ord('s'):
                    self.emergency_stop()
                        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of all components. Safe to call multiple times."""
        if not getattr(self, '_shutdown_done', False):
            self._shutdown_done = True
            print("\nShutting down...")
            
            # Play shutdown audio and wait for it to finish
            play_audio(AUDIO_SHUTTING_DOWN, block=True)
            
            self.running = False
            
            if self.motor and self.motor.connected:
                self.motor.go_home()
                self.motor.disconnect()
            
            if self.tracker:
                self.tracker.stop()
            
            self._stop_key_listener()
            
            cv2.destroyAllWindows()
            
            print("Shutdown complete.")


def run_calibration(
    mode: str = "auto",
    grid_cols: int = 4,
    grid_rows: int = 3,
):
    """
    Run camera/gantry calibration.

    Modes:
      - auto: blue-marker detection
      - manual: click 4 corners (optional motor assist)
      - motor-sweep: move gantry through a grid, click magnet at each stop
    """
    # Lazy-import calibration only when running standalone calibration CLI
    from calibration import CalibrationWizard, AutoCalibrator, MotorSweepCalibrator

    if mode == "manual":
        print("Starting manual calibration wizard...")
        motor = GRBLController()
        if motor.connect():
            motor.soft_home()
        else:
            motor = None
        
        wizard = CalibrationWizard()
        wizard.run_wizard(grbl_controller=motor)
        
        if motor:
            motor.disconnect()
    elif mode == "motor-sweep":
        print("Starting motor sweep calibration...")
        motor = GRBLController()
        if not motor.connect():
            print("ERROR: Motor sweep calibration requires motor connection.")
            return
        try:
            motor.soft_home()
            calibrator = MotorSweepCalibrator()
            calibrator.run(
                grbl_controller=motor,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
            )
        finally:
            motor.disconnect()
    else:
        print("Starting auto-calibration (blue marker detection)...")
        calibrator = AutoCalibrator()
        result = calibrator.run(show_preview=True)
        if result is None:
            print("\nAuto-calibration failed. You can try:")
            print("  python main.py --calibrate          (auto, try again)")
            print("  python main.py --calibrate-manual   (interactive click wizard)")
            return
    
    print("\nCalibration complete!")
    print("You can now run: python main.py")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Magic Table - Voice-Controlled Object Retrieval"
    )
    parser.add_argument(
        "--no-voice", 
        action="store_true",
        help="Disable voice control"
    )
    parser.add_argument(
        "--no-motor",
        action="store_true", 
        help="Disable motor control (vision only)"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run auto-calibration (detect 4 blue markers)"
    )
    parser.add_argument(
        "--calibrate-manual",
        action="store_true",
        help="Run manual calibration wizard (click corners)"
    )
    parser.add_argument(
        "--calibrate-motor",
        action="store_true",
        help="Run motor sweep calibration (moves gantry across a point grid)"
    )
    parser.add_argument(
        "--cal-grid-x",
        type=int,
        default=4,
        help="Motor sweep calibration grid columns (default: 4)"
    )
    parser.add_argument(
        "--cal-grid-y",
        type=int,
        default=3,
        help="Motor sweep calibration grid rows (default: 3)"
    )
    parser.add_argument(
        "--test-vision",
        action="store_true",
        help="Test vision system only"
    )
    parser.add_argument(
        "--test-voice",
        action="store_true",
        help="Test voice recognition only"
    )
    parser.add_argument(
        "--test-motor",
        action="store_true",
        help="Test motor control only"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Object label to detect (e.g. 'keys', 'phone'). Prompted if not given."
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable the web API server (enabled by default on port 5050)"
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=5050,
        help="Port for the web API server (default: 5050)"
    )
    
    args = parser.parse_args()
    
    # Handle special test modes
    if args.calibrate:
        run_calibration(mode="auto")
        return
    
    if args.calibrate_manual:
        run_calibration(mode="manual")
        return

    if args.calibrate_motor:
        run_calibration(
            mode="motor-sweep",
            grid_cols=args.cal_grid_x,
            grid_rows=args.cal_grid_y,
        )
        return
    
    if args.test_vision:
        from object_tracker import test_tracker
        test_tracker()
        return
    
    if args.test_voice:
        from voice_control import test_voice
        test_voice()
        return
    
    if args.test_motor:
        from grbl_controller import test_movement
        test_movement()
        return
    
    # Optional target label (for 'V' key manual detection)
    target_label = args.label
    if target_label:
        print(f"\nManual target label: {target_label} (press 'V' to detect)")
    
    # Run main system
    table = MagicTable(
        enable_voice=not args.no_voice,
        enable_motor=not args.no_motor,
        target_label=target_label,
    )
    
    if table.initialize():
        # Start the web API server in a background thread (on by default)
        if not args.no_web:
            from web_server import start_web_server
            start_web_server(magic_table=table, port=args.web_port)

        try:
            table.run()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            table.shutdown()
    else:
        print("Failed to initialize system")
        sys.exit(1)


if __name__ == "__main__":
    main()
