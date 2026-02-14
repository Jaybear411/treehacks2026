"""
Object Tracker - Vision-based object detection and tracking.

This module wraps the Hugging Face zero-shot object detection pipeline
and provides a clean interface for the main controller.

Features:
- Zero-shot object detection using OWLv2
- On-demand detection (press 'V' to scan)
- Persists detection results until next scan
- Single-label detection for faster inference
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from config import DETECTION, CAMERA


@dataclass
class TrackedObject:
    """A tracked object with position."""
    label: str
    center_x: float  # Pixel X
    center_y: float  # Pixel Y
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    last_seen: float = field(default_factory=time.time)


class ObjectTracker:
    """
    Vision-based object detection and tracking.
    
    Detection is on-demand (call run_detection() or press 'V' in GUI).
    Results persist until the next detection.
    
    Usage:
        tracker = ObjectTracker()
        tracker.start()
        
        while True:
            frame = tracker.capture_frame()  # Just grabs camera frame
            
            if user_pressed_v:
                tracker.run_detection()  # Runs VLM on current frame
            
            display = tracker.draw_detections(frame)
    """
    
    def __init__(self, prompts: List[str] = None, model_name: str = None):
        self.prompts = prompts or DETECTION.prompts
        self.model_name = model_name or DETECTION.model_name
        self.score_threshold = DETECTION.score_threshold
        self.max_detect_dim = DETECTION.max_detect_dim
        
        # Detection pipeline (lazy loaded)
        self._detector = None
        
        # Pick the best available device
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        
        # Camera
        self._cap: Optional[cv2.VideoCapture] = None
        self._camera_index = CAMERA.index
        
        # Tracked objects (persist until next detection)
        self._objects: Dict[str, TrackedObject] = {}
        
        # Last frame
        self._last_frame: Optional[np.ndarray] = None
        
        # Detection status
        self._last_detection_time: Optional[float] = None
        self._detection_in_progress = False
    
    def _load_detector(self):
        """Lazy-load the detection model."""
        if self._detector is not None:
            return
        
        print(f"Loading detection model: {self.model_name}")
        print(f"Using device: {self._device}")
        print("This may take a moment on first run...")
        
        from transformers import pipeline
        
        # Map device string to pipeline device arg
        if self._device == "mps":
            device_arg = "mps"
        elif self._device == "cuda":
            device_arg = 0
        else:
            device_arg = -1
        
        # Use float16 on GPU devices for faster inference
        dtype = torch.float16 if self._device in ("mps", "cuda") else None
        
        self._detector = pipeline(
            task="zero-shot-object-detection",
            model=self.model_name,
            device=device_arg,
            dtype=dtype,
        )
        
        print("Detection model loaded.")
    
    def start(self, camera_index: int = None) -> bool:
        """
        Start the camera and load the detection model.
        
        Returns True if successful.
        """
        camera_index = camera_index or self._camera_index
        
        # Load model
        self._load_detector()
        
        # Open camera
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            print(f"ERROR: Could not open camera {camera_index}")
            return False
        
        # Set resolution if supported
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.height)
        
        print(f"Camera opened: {camera_index}")
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Resolution: {actual_w}x{actual_h}")
        
        return True
    
    def stop(self):
        """Stop the camera."""
        if self._cap:
            self._cap.release()
            self._cap = None
        print("Camera stopped.")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera (NO detection).
        
        Returns the frame (or None if capture failed).
        """
        if not self._cap or not self._cap.isOpened():
            return None
        
        ret, frame = self._cap.read()
        if not ret:
            return None
        
        self._last_frame = frame
        return frame
    
    def run_detection(self, target_label: str = None) -> List[TrackedObject]:
        """
        Run object detection on the current frame for a single label.
        
        Args:
            target_label: The specific object label to detect. If None,
                          uses the first prompt in self.prompts.
        
        Returns list of detected objects.
        """
        if self._last_frame is None:
            print("No frame available for detection")
            return []
        
        if self._detector is None:
            print("Detector not loaded")
            return []
        
        # Use a single label for fast inference
        label_to_find = target_label or (self.prompts[0] if self.prompts else None)
        if not label_to_find:
            print("No label specified for detection")
            return []
        
        self._detection_in_progress = True
        print("\n" + "="*50)
        print("Running VLM detection...")
        print(f"Looking for: {label_to_find}")
        print("="*50)
        
        # Downscale image for faster inference
        h_orig, w_orig = self._last_frame.shape[:2]
        scale = min(self.max_detect_dim / w_orig, self.max_detect_dim / h_orig, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w_orig * scale), int(h_orig * scale)
            resized = cv2.resize(self._last_frame, (new_w, new_h))
        else:
            resized = self._last_frame
            scale = 1.0
        
        inv_scale = 1.0 / scale
        
        # Convert to RGB PIL Image for the detector
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_image = Image.fromarray(rgb)
        
        # Run detection with a single candidate label
        start_time = time.time()
        try:
            results = self._detector(
                pil_image,
                candidate_labels=[label_to_find]
            )
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            self._detection_in_progress = False
            return []
        
        elapsed = time.time() - start_time
        print(f"Detection completed in {elapsed:.2f}s")
        print(f"Raw results count: {len(results) if results else 0}")
        
        # Debug: print raw results
        if not results:
            print("No results returned from model")
        else:
            for i, r in enumerate(results):
                score = r.get('score', 0)
                print(f"  [{i}] label={r.get('label')}, score={score:.3f}, box={r.get('box')}")
        
        # Clear previous detections
        self._objects.clear()
        
        # Find best detection per label
        best: Dict[str, dict] = {}
        for r in results:
            label = r.get("label", "")
            score = float(r.get("score", 0.0))
            if score < self.score_threshold:
                continue
            if label not in best or score > float(best[label]["score"]):
                best[label] = r
        
        now = time.time()
        
        # Process detections
        for label, r in best.items():
            box = r["box"]
            
            # Debug the box format
            print(f"\nProcessing '{label}':")
            print(f"  Box type: {type(box)}")
            print(f"  Box value: {box}")
            
            # Extract bbox coordinates - handle multiple formats
            try:
                if isinstance(box, dict):
                    # Dict format: {"xmin": ..., "ymin": ..., "xmax": ..., "ymax": ...}
                    print(f"  Box keys: {box.keys()}")
                    if "xmin" in box:
                        x1 = float(box["xmin"])
                        y1 = float(box["ymin"])
                        x2 = float(box["xmax"])
                        y2 = float(box["ymax"])
                    else:
                        x1 = float(box["x1"])
                        y1 = float(box["y1"])
                        x2 = float(box["x2"])
                        y2 = float(box["y2"])
                elif isinstance(box, (list, tuple)):
                    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                elif hasattr(box, 'tolist'):
                    box_list = box.tolist()
                    x1, y1, x2, y2 = float(box_list[0]), float(box_list[1]), float(box_list[2]), float(box_list[3])
                elif hasattr(box, 'cpu'):
                    box_list = box.cpu().tolist()
                    x1, y1, x2, y2 = float(box_list[0]), float(box_list[1]), float(box_list[2]), float(box_list[3])
                else:
                    print(f"  ERROR: Unknown box format: {type(box)}")
                    continue
                    
            except Exception as e:
                print(f"  ERROR extracting box: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Scale bounding box back to original image coordinates
            x1 *= inv_scale
            y1 *= inv_scale
            x2 *= inv_scale
            y2 *= inv_scale
            
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            print(f"  Extracted (scaled to original): ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
            print(f"  Center: ({cx:.1f}, {cy:.1f})")
            
            obj = TrackedObject(
                label=label,
                center_x=cx,
                center_y=cy,
                confidence=float(r["score"]),
                bbox=(x1, y1, x2, y2),
                last_seen=now
            )
            self._objects[label] = obj
        
        self._last_detection_time = now
        self._detection_in_progress = False
        
        print("\n" + "="*50)
        print(f"Detected {len(self._objects)} objects:")
        for label, obj in self._objects.items():
            print(f"  - {label}: ({obj.center_x:.0f}, {obj.center_y:.0f}) conf={obj.confidence:.2f}")
        print("="*50 + "\n")
        
        return list(self._objects.values())
    
    def get_objects(self) -> Dict[str, TrackedObject]:
        """Get currently tracked objects."""
        return self._objects.copy()
    
    def get_object(self, label: str) -> Optional[TrackedObject]:
        """Get a specific object by label."""
        return self._objects.get(label)
    
    def get_object_position(self, label: str) -> Optional[Tuple[float, float]]:
        """Get the pixel position of an object."""
        obj = self._objects.get(label)
        if obj:
            return (obj.center_x, obj.center_y)
        return None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the last captured frame."""
        return self._last_frame
    
    def draw_detections(self, frame: np.ndarray = None) -> np.ndarray:
        """
        Draw detection overlays on a frame.
        
        Args:
            frame: Frame to draw on (uses last frame if None)
            
        Returns:
            Frame with detection overlays
        """
        if frame is None:
            frame = self._last_frame
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        frame = frame.copy()
        
        # Draw each detected object
        for label, obj in self._objects.items():
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            cx, cy = int(obj.center_x), int(obj.center_y)
            
            # Draw bounding box (green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point (red)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
            
            # Draw label background
            text = f"{label} {obj.confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(frame, text, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw coordinates below box
            coord_text = f"({cx}, {cy})"
            cv2.putText(frame, coord_text, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw detection status
        if self._detection_in_progress:
            cv2.putText(frame, "DETECTING...", (frame.shape[1]//2 - 80, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        return frame
    
    def find_object(self, label: str) -> Optional[TrackedObject]:
        """
        Find an object by label (case-insensitive, partial match).
        
        Returns the best matching TrackedObject or None.
        """
        label_lower = label.lower()
        
        # Exact match first
        for obj_label, obj in self._objects.items():
            if obj_label.lower() == label_lower:
                return obj
        
        # Partial match
        for obj_label, obj in self._objects.items():
            if label_lower in obj_label.lower() or obj_label.lower() in label_lower:
                return obj
        
        return None
    
    def list_visible_objects(self) -> List[str]:
        """Get list of currently visible object labels."""
        return list(self._objects.keys())
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def test_tracker():
    """Interactive test of object tracking."""
    print("\n" + "="*50)
    print("Object Tracker Test")
    print("="*50)
    
    # Prompt for a single label
    print(f"Available objects: {', '.join(DETECTION.prompts)}")
    target_label = input("Enter the object to detect: ").strip()
    if not target_label:
        target_label = DETECTION.prompts[0]
        print(f"Defaulting to: {target_label}")
    
    print(f"\nPress 'V' to run VLM detection for '{target_label}'")
    print("Press 'Q' to quit")
    print("="*50 + "\n")
    
    tracker = ObjectTracker()
    tracker.start()
    
    try:
        while True:
            # Just capture frame (no detection)
            frame = tracker.capture_frame()
            if frame is None:
                continue
            
            # Draw any existing detections
            display = tracker.draw_detections(frame)
            
            # Draw instructions
            cv2.putText(display, f"Press 'V' to detect '{target_label}'", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show object list if any
            objects = tracker.list_visible_objects()
            if objects:
                text = "Detected: " + ", ".join(objects)
                cv2.putText(display, text, (10, display.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Object Tracker", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                # Run detection for the target label
                tracker.run_detection(target_label=target_label)
                
    finally:
        tracker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_tracker()
