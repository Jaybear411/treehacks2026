"""
Calibration - Pixel-to-Physical coordinate transformation.

This module handles:
- Camera calibration (mapping pixel coordinates to physical mm)
- Perspective correction (camera may not be perfectly overhead)
- Interactive calibration wizard

The calibration uses a homography matrix to map between:
- Pixel space: (px, py) from camera image
- Physical space: (x, y) in mm on the gantry

Calibration Process:
1. Move the magnet to known corners of the gantry
2. Click the corresponding point in the camera view
3. System computes transformation matrix
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import GANTRY, CAMERA, CALIBRATION, MARKER


@dataclass
class CalibrationPoint:
    """A calibration point with pixel and physical coordinates."""
    pixel_x: float
    pixel_y: float
    physical_x: float
    physical_y: float


class CoordinateTransformer:
    """
    Transforms coordinates between pixel space and physical space.
    
    Uses homography (perspective transform) to handle camera angle/distortion.
    """
    
    CALIBRATION_FILE = "calibration_data.json"
    
    def __init__(self):
        self.calibration_points: List[CalibrationPoint] = []
        self._homography: Optional[np.ndarray] = None
        self._inverse_homography: Optional[np.ndarray] = None
        
        # Simple linear calibration (fallback)
        self._scale_x = GANTRY.width_mm / CAMERA.width
        self._scale_y = GANTRY.height_mm / CAMERA.height
        self._offset_x = 0.0
        self._offset_y = 0.0
        
    def add_calibration_point(self, pixel: Tuple[float, float], 
                              physical: Tuple[float, float]):
        """Add a calibration point."""
        self.calibration_points.append(CalibrationPoint(
            pixel_x=pixel[0],
            pixel_y=pixel[1],
            physical_x=physical[0],
            physical_y=physical[1]
        ))
    
    def compute_transform(self) -> bool:
        """
        Compute the transformation matrix from calibration points.
        
        Requires at least 4 points for homography.
        Falls back to linear approximation with fewer points.
        """
        n = len(self.calibration_points)
        
        if n == 0:
            print("No calibration points. Using default linear mapping.")
            return False
        
        if n < 4:
            print(f"Only {n} points. Using linear approximation (4+ recommended).")
            self._compute_linear_transform()
            return True
        
        # Compute homography with 4+ points
        src_pts = np.array([
            [p.pixel_x, p.pixel_y] for p in self.calibration_points
        ], dtype=np.float32)
        
        dst_pts = np.array([
            [p.physical_x, p.physical_y] for p in self.calibration_points
        ], dtype=np.float32)
        
        # Compute homography matrix
        self._homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        
        if self._homography is not None:
            self._inverse_homography = np.linalg.inv(self._homography)
            print("Homography calibration complete.")
            return True
        else:
            print("Homography computation failed. Using linear approximation.")
            self._compute_linear_transform()
            return True
    
    def _compute_linear_transform(self):
        """Compute simple linear scale + offset transformation."""
        if len(self.calibration_points) == 0:
            return
        
        # Average the scale factors from all points
        scale_x_sum = 0.0
        scale_y_sum = 0.0
        offset_x_sum = 0.0
        offset_y_sum = 0.0
        
        for p in self.calibration_points:
            if p.pixel_x != 0:
                scale_x_sum += p.physical_x / p.pixel_x
            if p.pixel_y != 0:
                scale_y_sum += p.physical_y / p.pixel_y
        
        n = len(self.calibration_points)
        self._scale_x = scale_x_sum / n if n > 0 else self._scale_x
        self._scale_y = scale_y_sum / n if n > 0 else self._scale_y
    
    def pixel_to_physical(self, px: float, py: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to physical coordinates (mm).
        
        Args:
            px: X pixel coordinate
            py: Y pixel coordinate
            
        Returns:
            (x_mm, y_mm): Physical coordinates in mm
        """
        if self._homography is not None:
            # Use homography transform
            pt = np.array([[[px, py]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self._homography)
            return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])
        else:
            # Linear transform
            x = px * self._scale_x + self._offset_x
            y = py * self._scale_y + self._offset_y
            return x, y
    
    def physical_to_pixel(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """
        Convert physical coordinates (mm) to pixel coordinates.
        
        Args:
            x_mm: X physical coordinate in mm
            y_mm: Y physical coordinate in mm
            
        Returns:
            (px, py): Pixel coordinates
        """
        if self._inverse_homography is not None:
            pt = np.array([[[x_mm, y_mm]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self._inverse_homography)
            return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])
        else:
            px = (x_mm - self._offset_x) / self._scale_x
            py = (y_mm - self._offset_y) / self._scale_y
            return px, py
    
    def save_calibration(self, filepath: str = None):
        """Save calibration data to file."""
        filepath = filepath or self.CALIBRATION_FILE
        
        data = {
            "points": [
                {
                    "pixel": [p.pixel_x, p.pixel_y],
                    "physical": [p.physical_x, p.physical_y]
                }
                for p in self.calibration_points
            ],
            "homography": self._homography.tolist() if self._homography is not None else None,
            "linear": {
                "scale_x": self._scale_x,
                "scale_y": self._scale_y,
                "offset_x": self._offset_x,
                "offset_y": self._offset_y
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str = None) -> bool:
        """Load calibration data from file."""
        filepath = filepath or self.CALIBRATION_FILE
        
        if not os.path.exists(filepath):
            print(f"Calibration file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load points
            self.calibration_points = [
                CalibrationPoint(
                    pixel_x=p["pixel"][0],
                    pixel_y=p["pixel"][1],
                    physical_x=p["physical"][0],
                    physical_y=p["physical"][1]
                )
                for p in data.get("points", [])
            ]
            
            # Load homography
            if data.get("homography"):
                self._homography = np.array(data["homography"], dtype=np.float32)
                self._inverse_homography = np.linalg.inv(self._homography)
            
            # Load linear params
            linear = data.get("linear", {})
            self._scale_x = linear.get("scale_x", self._scale_x)
            self._scale_y = linear.get("scale_y", self._scale_y)
            self._offset_x = linear.get("offset_x", self._offset_x)
            self._offset_y = linear.get("offset_y", self._offset_y)
            
            print(f"Calibration loaded from {filepath}")
            print(f"  {len(self.calibration_points)} calibration points")
            return True
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False


class CalibrationWizard:
    """
    Interactive calibration wizard using OpenCV GUI.
    
    Guides user through clicking four corners while the magnet
    is positioned at known physical locations.
    """
    
    # Default calibration corners (physical coordinates in mm)
    DEFAULT_CORNERS = [
        (GANTRY.margin_mm, GANTRY.margin_mm),                    # Bottom-left
        (GANTRY.width_mm - GANTRY.margin_mm, GANTRY.margin_mm),  # Bottom-right
        (GANTRY.width_mm - GANTRY.margin_mm, GANTRY.height_mm - GANTRY.margin_mm),  # Top-right
        (GANTRY.margin_mm, GANTRY.height_mm - GANTRY.margin_mm),  # Top-left
    ]
    
    CORNER_NAMES = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]
    
    def __init__(self, camera_index: int = None):
        self.camera_index = camera_index or CAMERA.index
        self.transformer = CoordinateTransformer()
        
        self._clicked_point: Optional[Tuple[int, int]] = None
        self._current_frame: Optional[np.ndarray] = None
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse click handler for calibration."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicked_point = (x, y)
    
    def run_wizard(self, grbl_controller=None) -> CoordinateTransformer:
        """
        Run the interactive calibration wizard.
        
        Args:
            grbl_controller: Optional GRBLController to auto-move magnet
            
        Returns:
            Configured CoordinateTransformer
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
        
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self._mouse_callback)
        
        print("\n" + "="*50)
        print("CALIBRATION WIZARD")
        print("="*50)
        print("You will click four corners of the gantry area.")
        print("For each corner, the system will show where to position")
        print("the magnet, then you click that spot in the camera view.")
        print("\nPress 'q' to cancel, 's' to skip to next corner.")
        print("="*50 + "\n")
        
        for i, (phys_x, phys_y) in enumerate(self.DEFAULT_CORNERS):
            corner_name = self.CORNER_NAMES[i]
            
            print(f"\nCorner {i+1}/4: {corner_name}")
            print(f"Position magnet at: ({phys_x:.0f}, {phys_y:.0f}) mm")
            
            # Auto-move if controller available
            if grbl_controller and grbl_controller.connected:
                print("Moving magnet to position...")
                grbl_controller.move_to(phys_x, phys_y)
            
            print("Click the magnet position in the camera view...")
            
            self._clicked_point = None
            
            while self._clicked_point is None:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                self._current_frame = frame.copy()
                
                # Draw instructions
                text = f"Click {corner_name} ({phys_x:.0f}, {phys_y:.0f}) mm"
                cv2.putText(frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Corner {i+1}/4 - Press 'q' to cancel", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw previously calibrated points
                for j, pt in enumerate(self.transformer.calibration_points):
                    cv2.circle(frame, (int(pt.pixel_x), int(pt.pixel_y)), 
                              8, (0, 255, 0), -1)
                    cv2.putText(frame, self.CORNER_NAMES[j], 
                               (int(pt.pixel_x) + 10, int(pt.pixel_y)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow("Calibration", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Calibration cancelled.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return self.transformer
                elif key == ord('s'):
                    print(f"Skipping {corner_name}")
                    break
            
            if self._clicked_point:
                px, py = self._clicked_point
                self.transformer.add_calibration_point(
                    pixel=(px, py),
                    physical=(phys_x, phys_y)
                )
                print(f"  Recorded: pixel ({px}, {py}) -> physical ({phys_x:.0f}, {phys_y:.0f})")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Compute transform
        self.transformer.compute_transform()
        
        # Save calibration
        self.transformer.save_calibration()
        
        print("\nCalibration complete!")
        return self.transformer
    
    def quick_calibrate(self) -> CoordinateTransformer:
        """
        Quick calibration using assumed camera-aligned coordinates.
        
        Assumes camera is mounted directly overhead and aligned with gantry.
        Uses simple linear scaling based on camera resolution.
        """
        print("Quick calibration (assuming overhead camera alignment)")
        
        # Map camera corners to gantry corners
        # Assumes: camera (0,0) = gantry top-left, camera max = gantry bottom-right
        # Note: Camera Y is typically inverted from physical Y
        
        self.transformer.add_calibration_point(
            pixel=(0, 0),
            physical=(0, GANTRY.height_mm)  # Top-left
        )
        self.transformer.add_calibration_point(
            pixel=(CAMERA.width, 0),
            physical=(GANTRY.width_mm, GANTRY.height_mm)  # Top-right
        )
        self.transformer.add_calibration_point(
            pixel=(CAMERA.width, CAMERA.height),
            physical=(GANTRY.width_mm, 0)  # Bottom-right
        )
        self.transformer.add_calibration_point(
            pixel=(0, CAMERA.height),
            physical=(0, 0)  # Bottom-left
        )
        
        self.transformer.compute_transform()
        self.transformer.save_calibration()
        
        print("Quick calibration complete!")
        return self.transformer


class MotorSweepCalibrator:
    """
    Motor-driven calibration that sweeps across the board in a grid.

    Workflow:
    1. Gantry auto-moves to each known physical point.
    2. User clicks the magnet location in camera view.
    3. System computes and saves a homography using all recorded points.
    """

    def __init__(self, camera_index: int = None):
        self.camera_index = camera_index if camera_index is not None else CAMERA.index
        self.transformer = CoordinateTransformer()
        self._clicked_point: Optional[Tuple[int, int]] = None

    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse click handler for calibration capture."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicked_point = (x, y)

    @staticmethod
    def _generate_grid_points(
        grid_cols: int,
        grid_rows: int,
    ) -> List[Tuple[float, float]]:
        """
        Generate serpentine traversal points within gantry limits.
        """
        cols = max(2, grid_cols)
        rows = max(2, grid_rows)
        min_x, max_x = GANTRY.min_x, GANTRY.max_x
        min_y, max_y = GANTRY.min_y, GANTRY.max_y

        xs = np.linspace(min_x, max_x, cols).tolist()
        ys = np.linspace(min_y, max_y, rows).tolist()

        points: List[Tuple[float, float]] = []
        for r, y in enumerate(ys):
            row_xs = xs if (r % 2 == 0) else list(reversed(xs))
            for x in row_xs:
                points.append((float(x), float(y)))
        return points

    def run(
        self,
        grbl_controller,
        grid_cols: int = 4,
        grid_rows: int = 3,
        settle_seconds: float = 0.35,
    ) -> CoordinateTransformer:
        """
        Run motor sweep calibration using a connected GRBL controller.
        """
        if grbl_controller is None or not grbl_controller.connected:
            raise RuntimeError("Motor sweep calibration requires a connected GRBL controller.")

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")

        cv2.namedWindow("Motor Sweep Calibration")
        cv2.setMouseCallback("Motor Sweep Calibration", self._mouse_callback)

        points = self._generate_grid_points(grid_cols=grid_cols, grid_rows=grid_rows)

        print("\n" + "=" * 60)
        print("MOTOR SWEEP CALIBRATION")
        print("=" * 60)
        print(f"Grid: {max(2, grid_cols)} x {max(2, grid_rows)} ({len(points)} points)")
        print("The gantry will move point-to-point. Click the magnet each time.")
        print("Controls: click = record, 's' = skip point, 'q' = cancel")
        print("=" * 60 + "\n")

        for idx, (phys_x, phys_y) in enumerate(points, start=1):
            print(f"[{idx}/{len(points)}] Moving to ({phys_x:.1f}, {phys_y:.1f}) mm...")
            moved = grbl_controller.move_to(phys_x, phys_y, wait=True)
            if not moved:
                print("  Move failed or blocked by limit. Skipping this point.")
                continue

            # Let vibration settle before clicking.
            cv2.waitKey(int(max(0.0, settle_seconds) * 1000))
            self._clicked_point = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Draw overlay
                cv2.putText(
                    frame,
                    f"Point {idx}/{len(points)}  Physical: ({phys_x:.1f}, {phys_y:.1f}) mm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Click magnet | 's' skip | 'q' cancel",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

                # Draw already-recorded points
                for p in self.transformer.calibration_points:
                    cv2.circle(frame, (int(p.pixel_x), int(p.pixel_y)), 5, (0, 255, 255), -1)

                cv2.imshow("Motor Sweep Calibration", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Calibration cancelled by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    self.transformer.compute_transform()
                    self.transformer.save_calibration()
                    return self.transformer

                if key == ord("s"):
                    print("  Skipped.")
                    break

                if self._clicked_point is not None:
                    px, py = self._clicked_point
                    self.transformer.add_calibration_point(
                        pixel=(px, py),
                        physical=(phys_x, phys_y),
                    )
                    print(
                        "  Recorded: "
                        f"pixel ({px}, {py}) -> physical ({phys_x:.1f}, {phys_y:.1f})"
                    )
                    break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nCaptured {len(self.transformer.calibration_points)} calibration points.")
        self.transformer.compute_transform()
        self.transformer.save_calibration()
        print("Motor sweep calibration complete.")
        return self.transformer


class AutoCalibrator:
    """
    Automatic calibration using blue marker dots.
    
    Place 4 blue dots (tape, stickers, etc.) at the known gantry corners.
    This class detects them via HSV color filtering, assigns each to the
    nearest expected corner, and computes the homography.
    
    Much faster and more repeatable than the manual click wizard.
    """
    
    def __init__(self, camera_index: int = None):
        self.camera_index = camera_index if camera_index is not None else CAMERA.index
        self.transformer = CoordinateTransformer()
        
        # HSV thresholds from config
        self._hsv_low = np.array([MARKER.hue_low, MARKER.sat_low, MARKER.val_low])
        self._hsv_high = np.array([MARKER.hue_high, MARKER.sat_high, MARKER.val_high])
    
    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------
    
    def detect_markers(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        """
        Detect blue dot markers in a single frame.
        
        Returns list of (cx, cy) pixel centers for each detected marker,
        sorted top-left first then clockwise.
        """
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (MARKER.blur_kernel, MARKER.blur_kernel), 0)
        
        # Convert to HSV and threshold for blue
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_low, self._hsv_high)
        
        # Morphological cleanup — close small holes, then remove small specks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates: List[Tuple[float, float]] = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MARKER.min_area or area > MARKER.max_area:
                continue
            
            # Circularity check
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < MARKER.min_circularity:
                continue
            
            # Centroid via moments
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            candidates.append((cx, cy))
        
        return candidates
    
    def detect_markers_averaged(self, cap: cv2.VideoCapture) -> List[Tuple[float, float]]:
        """
        Detect markers over several frames and average the positions
        for more stable results.
        """
        all_detections: List[List[Tuple[float, float]]] = []
        
        for _ in range(MARKER.detection_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            markers = self.detect_markers(frame)
            if len(markers) >= 4:
                # Keep only the 4 largest-area-ish (we already filtered),
                # but ensure we have exactly 4 by sorting and trimming
                all_detections.append(markers[:4] if len(markers) > 4 else markers)
        
        if not all_detections:
            return []
        
        # Use the detection set that appeared most often with exactly 4 markers
        # For simplicity: just use the last good detection set and average
        # positions across all frames that had 4 markers.
        # 
        # Pair points across frames by nearest-neighbor matching to the
        # first good frame's ordering.
        
        reference = self._sort_corners(all_detections[0])
        if len(reference) != 4:
            return reference
        
        accum = [list(pt) for pt in reference]
        count = 1
        
        for det in all_detections[1:]:
            sorted_det = self._sort_corners(det)
            if len(sorted_det) != 4:
                continue
            # Match to reference by nearest neighbor
            matched = self._match_to_reference(reference, sorted_det)
            if matched is None:
                continue
            for i in range(4):
                accum[i][0] += matched[i][0]
                accum[i][1] += matched[i][1]
            count += 1
        
        averaged = [(a[0] / count, a[1] / count) for a in accum]
        return averaged
    
    # ------------------------------------------------------------------
    # Corner sorting / matching
    # ------------------------------------------------------------------
    
    @staticmethod
    def _sort_corners(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Sort detected points into canonical order:
        [top-left, top-right, bottom-right, bottom-left]
        
        Uses centroid-relative angle sorting (clockwise from top-left).
        """
        if len(points) < 4:
            return points
        
        pts = list(points[:4])
        
        # Centroid
        cx = sum(p[0] for p in pts) / 4.0
        cy = sum(p[1] for p in pts) / 4.0
        
        # Separate into top (lower y) and bottom (higher y)
        # In image coords, y increases downward
        top = sorted([p for p in pts if p[1] < cy], key=lambda p: p[0])
        bottom = sorted([p for p in pts if p[1] >= cy], key=lambda p: p[0])
        
        # Handle edge case where the split isn't clean 2-2
        if len(top) != 2 or len(bottom) != 2:
            # Fallback: sort by sum/diff for corner assignment
            # top-left has smallest x+y, bottom-right has largest x+y
            # top-right has smallest y-x, bottom-left has largest y-x
            s = sorted(pts, key=lambda p: p[0] + p[1])
            d = sorted(pts, key=lambda p: p[1] - p[0])
            return [s[0], d[0], s[3], d[3]]  # TL, TR, BR, BL
        
        tl, tr = top[0], top[1]
        bl, br = bottom[0], bottom[1]
        
        return [tl, tr, br, bl]
    
    @staticmethod
    def _match_to_reference(
        reference: List[Tuple[float, float]],
        candidates: List[Tuple[float, float]],
    ) -> Optional[List[Tuple[float, float]]]:
        """Match candidate points to reference points by nearest neighbor."""
        if len(candidates) != len(reference):
            return None
        
        matched = [None] * len(reference)
        used = set()
        
        for i, ref in enumerate(reference):
            best_dist = float("inf")
            best_j = -1
            for j, cand in enumerate(candidates):
                if j in used:
                    continue
                dist = (ref[0] - cand[0]) ** 2 + (ref[1] - cand[1]) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j >= 0:
                matched[i] = candidates[best_j]
                used.add(best_j)
        
        if any(m is None for m in matched):
            return None
        return matched
    
    # ------------------------------------------------------------------
    # Full calibration pipeline
    # ------------------------------------------------------------------
    
    def run(self, show_preview: bool = True) -> Optional[CoordinateTransformer]:
        """
        Run automatic calibration.
        
        1. Open camera and detect 4 blue markers
        2. Assign each to the nearest gantry corner
        3. Compute homography and save
        
        Args:
            show_preview: If True, show a live OpenCV window during detection.
        
        Returns:
            Configured CoordinateTransformer, or None on failure.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"ERROR: Could not open camera {self.camera_index}")
            return None
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.height)
        
        print("\n" + "=" * 60)
        print("AUTO-CALIBRATION — Blue Marker Detection")
        print("=" * 60)
        print("Place 4 blue dots at the gantry corners.")
        print("Press 'c' to capture and calibrate, 'q' to cancel.")
        print("=" * 60 + "\n")
        
        marker_pixels: List[Tuple[float, float]] = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Live detection for preview
            live_markers = self.detect_markers(frame)
            
            if show_preview:
                preview = frame.copy()
                self._draw_preview(preview, live_markers)
                cv2.imshow("Auto-Calibration", preview)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Calibration cancelled.")
                cap.release()
                if show_preview:
                    cv2.destroyAllWindows()
                return None
            
            if key == ord('c'):
                # Capture averaged detection
                print("Capturing markers (averaging over multiple frames)...")
                marker_pixels = self.detect_markers_averaged(cap)
                
                if len(marker_pixels) == 4:
                    print("Detected 4 markers!")
                    for i, (mx, my) in enumerate(marker_pixels):
                        print(f"  Marker {i}: pixel ({mx:.1f}, {my:.1f})")
                    break
                else:
                    count = len(marker_pixels)
                    print(f"Found {count} markers — need exactly 4. Adjust markers and try again.")
                    marker_pixels = []
        
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        if len(marker_pixels) != 4:
            print("Calibration failed: could not detect 4 markers.")
            return None
        
        # Map pixel corners → physical corners
        physical_corners = list(MARKER.marker_positions_mm)
        
        print("\nMapping markers to physical corners:")
        for i in range(4):
            px, py = marker_pixels[i]
            phys_x, phys_y = physical_corners[i]
            print(f"  pixel ({px:.1f}, {py:.1f}) -> physical ({phys_x:.0f}, {phys_y:.0f}) mm")
            self.transformer.add_calibration_point(
                pixel=(px, py),
                physical=(phys_x, phys_y),
            )
        
        # Compute homography
        success = self.transformer.compute_transform()
        if not success:
            print("Homography computation failed.")
            return None
        
        # Save
        self.transformer.save_calibration()
        
        print("\n" + "=" * 60)
        print("AUTO-CALIBRATION COMPLETE")
        print("=" * 60 + "\n")
        
        return self.transformer
    
    def run_headless(self, frame: np.ndarray) -> Optional[CoordinateTransformer]:
        """
        Run calibration on a single provided frame (no GUI).
        
        Useful for programmatic / startup calibration.
        """
        markers = self.detect_markers(frame)
        if len(markers) < 4:
            print(f"Auto-calibration: found {len(markers)} markers, need 4.")
            return None
        
        sorted_markers = self._sort_corners(markers[:4] if len(markers) > 4 else markers)
        physical_corners = list(MARKER.marker_positions_mm)
        
        for i in range(4):
            px, py = sorted_markers[i]
            phys_x, phys_y = physical_corners[i]
            self.transformer.add_calibration_point(
                pixel=(px, py),
                physical=(phys_x, phys_y),
            )
        
        self.transformer.compute_transform()
        self.transformer.save_calibration()
        print("Auto-calibration (headless) complete.")
        return self.transformer
    
    # ------------------------------------------------------------------
    # Preview drawing
    # ------------------------------------------------------------------
    
    @staticmethod
    def _draw_preview(
        frame: np.ndarray,
        markers: List[Tuple[float, float]],
    ):
        """Draw marker detection overlay on a frame."""
        n = len(markers)
        color = (0, 255, 0) if n == 4 else (0, 165, 255)  # green if 4, orange otherwise
        
        for i, (cx, cy) in enumerate(markers):
            cv2.circle(frame, (int(cx), int(cy)), 12, color, 3)
            cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            cv2.putText(
                frame, f"M{i}", (int(cx) + 15, int(cy) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
        
        status = f"Markers: {n}/4"
        status_color = (0, 255, 0) if n == 4 else (0, 0, 255)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        if n == 4:
            cv2.putText(frame, "Press 'c' to calibrate", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Adjust markers until 4 detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        cv2.putText(frame, "Press 'q' to cancel", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def test_transform():
    """Test coordinate transformation."""
    transformer = CoordinateTransformer()
    
    # Quick calibration
    wizard = CalibrationWizard()
    transformer = wizard.quick_calibrate()
    
    # Test some conversions
    test_pixels = [
        (0, 0),
        (640, 360),
        (1280, 720),
        (320, 180),
    ]
    
    print("\nTransformation Test:")
    print("-" * 50)
    for px, py in test_pixels:
        x, y = transformer.pixel_to_physical(px, py)
        px2, py2 = transformer.physical_to_pixel(x, y)
        print(f"Pixel ({px:4}, {py:4}) -> Physical ({x:6.1f}, {y:6.1f}) mm")
        print(f"  Round-trip: ({px2:.1f}, {py2:.1f})")


def run_calibration_wizard():
    """Run the full calibration wizard."""
    wizard = CalibrationWizard()
    transformer = wizard.run_wizard()
    return transformer


def run_auto_calibration():
    """Run the automatic blue-marker calibration."""
    calibrator = AutoCalibrator()
    return calibrator.run(show_preview=True)


def run_motor_sweep_calibration(grbl_controller, grid_cols: int = 4, grid_rows: int = 3):
    """Run motor-driven grid sweep calibration."""
    calibrator = MotorSweepCalibrator()
    return calibrator.run(
        grbl_controller=grbl_controller,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        wizard = CalibrationWizard()
        wizard.quick_calibrate()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test_transform()
    elif len(sys.argv) > 1 and sys.argv[1] == "manual":
        run_calibration_wizard()
    else:
        # Default: auto-calibration with blue markers
        run_auto_calibration()
