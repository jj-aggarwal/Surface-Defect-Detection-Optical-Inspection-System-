"""Defect feature extraction and classification (rule-based or learned)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import InspectionConfig

FEATURE_NAMES = ["area_px", "length_px", "width_px", "aspect", "solidity", "tortuosity",
                 "fill_ratio", "contrast", "peak_score", "mean_score", "wiggle"]


def _skeleton_stats(mask: np.ndarray):
    """Path length, tortuosity (path/chord) and normalised lateral wiggle of the skeleton."""
    from skimage.morphology import skeletonize
    sk = skeletonize(mask)
    ys, xs = np.nonzero(sk)
    n = len(xs)
    if n < 3:
        return float(max(n, 1)), 1.0, 0.0
    pts = np.stack([xs, ys], 1).astype(np.float64)
    pts -= pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    along, across = pts @ vt[0], pts @ vt[1]
    chord = float(along.max() - along.min()) + 1.0
    # 8-connected skeleton: diagonal steps are ~1.41 px, so scale the pixel count slightly
    path = n * 1.1
    return path, max(path / chord, 1.0), float(across.std() / chord)


def region_features(mask: np.ndarray, signed: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    """Geometric + photometric descriptors of one connected region (bool mask)."""
    m8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea) if cnts else None
    area = float(m8.sum())
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    rect_len, rect_wid = max(rw, rh, 1.0), max(min(rw, rh), 1.0)
    hull_area = cv2.contourArea(cv2.convexHull(cnt)) if len(cnt) >= 3 else area
    solidity = area / hull_area if hull_area > 0 else 1.0

    path, tort, wiggle = _skeleton_stats(mask)
    mean_width = max(area / path, 1.0)
    if mean_width * 4 < path:                       # elongated: measure along the skeleton
        length, width = path, mean_width
    else:                                           # compact: use the oriented bounding box
        length, width = rect_len, rect_wid

    ring = cv2.dilate(m8, np.ones((9, 9), np.uint8)) & (1 - cv2.dilate(m8, np.ones((3, 3), np.uint8)))
    ref = np.median(signed[ring > 0]) if ring.any() else 0.0
    ext = signed[mask]
    peak = float(ext[np.argmax(np.abs(ext - ref))] - ref)
    contrast = float(np.mean(ext) - ref)
    return {
        "area_px": area, "length_px": float(length), "width_px": float(width),
        "aspect": float(length / width), "solidity": float(solidity),
        "tortuosity": float(tort), "wiggle": float(wiggle),
        "fill_ratio": float(area / (rect_len * rect_wid)),
        "contrast": peak if abs(peak) > abs(contrast) else contrast,
        "peak_score": float(score[mask].max()), "mean_score": float(score[mask].mean()),
    }


def rule_classify(f: Dict[str, float], cfg: InspectionConfig) -> str:
    if f["area_px"] >= cfg.stain_min_area_px and f["aspect"] < 3.0 and f["fill_ratio"] > 0.5:
        return "stain"
    if f["aspect"] >= cfg.scratch_min_aspect:
        if f["tortuosity"] >= cfg.crack_min_tortuosity or f["wiggle"] >= cfg.crack_min_wiggle:
            return "crack"
        return "scratch"
    return "pit"


class LearnedClassifier:
    """Thin wrapper around a scikit-learn model trained on `FEATURE_NAMES`."""

    def __init__(self, model):
        self.model = model

    def predict(self, f: Dict[str, float]) -> str:
        x = np.array([[f[k] for k in FEATURE_NAMES]], np.float32)
        return str(self.model.predict(x)[0])

    def save(self, path: str | Path) -> None:
        import joblib
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedClassifier":
        import joblib
        return cls(joblib.load(path))
