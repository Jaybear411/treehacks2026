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

from config import GANTRY, CAMERA, CALIBRATION


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


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        wizard = CalibrationWizard()
        wizard.quick_calibrate()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test_transform()
    else:
        run_calibration_wizard()
