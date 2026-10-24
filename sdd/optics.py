"""Optical / metrology helpers: camera calibration, undistortion, pixel scale."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]      # (w, h)
    rms_error_px: float
    pixel_size_mm: float | None = None

    def undistort(self, img: np.ndarray) -> np.ndarray:
        return cv2.undistort(img, self.camera_matrix, self.dist_coeffs)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_size": list(self.image_size),
            "rms_error_px": self.rms_error_px,
            "pixel_size_mm": self.pixel_size_mm,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["camera_matrix"]), np.array(d["dist_coeffs"]),
                   tuple(d["image_size"]), d["rms_error_px"], d.get("pixel_size_mm"))


def find_checkerboard(gray: np.ndarray, pattern: Tuple[int, int]):
    """Return refined inner corners (N,1,2) or None. `pattern` = inner corners (cols, rows)."""
    ok, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    return cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), crit)


def calibrate_camera(images: Sequence[np.ndarray], pattern: Tuple[int, int] = (9, 6),
                     square_mm: float = 5.0) -> CameraCalibration:
    """Zhang calibration from several checkerboard views; also estimates mm/pixel."""
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm
    obj_pts, img_pts, size, steps = [], [], None, []
    for im in images:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        size = gray.shape[::-1]
        c = find_checkerboard(gray, pattern)
        if c is None:
            continue
        obj_pts.append(objp)
        img_pts.append(c)
        pts = c.reshape(pattern[1], pattern[0], 2)
        steps.append(np.median(np.linalg.norm(np.diff(pts, axis=1), axis=2)))
    if len(obj_pts) < 3:
        raise RuntimeError(f"Need >=3 usable checkerboard views, found {len(obj_pts)}")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    # Mean square size in pixels (fronto-parallel approximation) -> mm per pixel.
    px_mm = square_mm / float(np.max(steps))   # largest step ~ least foreshortened view
    return CameraCalibration(K, dist, size, float(rms), px_mm)


def pixel_size_from_reference(length_px: float, length_mm: float) -> float:
    """mm/pixel from a feature of known size (gauge block, ruler, fiducial pair)."""
    if length_px <= 0:
        raise ValueError("length_px must be > 0")
    return length_mm / length_px


def make_checkerboard(pattern=(9, 6), square_px=40, border=60) -> np.ndarray:
    """Render a flat checkerboard (used for tests / printable target)."""
    cols, rows = pattern[0] + 1, pattern[1] + 1
    img = np.full((rows * square_px + 2 * border, cols * square_px + 2 * border), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = border + r * square_px, border + c * square_px
                img[y:y + square_px, x:x + square_px] = 0
    return img
