"""End-to-end inspection pipeline: image -> defects -> measurements -> verdict."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .classify import LearnedClassifier, region_features, rule_classify
from .config import InspectionConfig
from .detect import segment
from .optics import CameraCalibration
from .preprocess import flat_field, to_gray_f32

COLORS = {"scratch": (0, 200, 255), "crack": (0, 0, 255), "pit": (255, 120, 0), "stain": (0, 220, 0)}


@dataclass
class Defect:
    id: int
    cls: str
    bbox: Tuple[int, int, int, int]          # x, y, w, h (px)
    centroid: Tuple[float, float]
    area_px: float
    area_mm2: float
    length_mm: float
    width_mm: float
    contrast: float                          # gray levels vs local surround (signed)
    score: float                             # peak robust z
    rejectable: bool = False


@dataclass
class InspectionResult:
    verdict: str                             # "OK" | "NG"
    defects: List[Defect]
    labels: np.ndarray = field(repr=False)
    score_map: np.ndarray = field(repr=False)
    image: np.ndarray = field(repr=False)
    elapsed_ms: float = 0.0
    reasons: List[str] = field(default_factory=list)

    @property
    def mask(self) -> np.ndarray:
        return self.labels > 0

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "n_defects": len(self.defects),
                "elapsed_ms": round(self.elapsed_ms, 2), "reasons": self.reasons,
                "defects": [asdict(d) for d in self.defects]}

    def overlay(self, draw_labels: bool = True) -> np.ndarray:
        vis = self.image.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        tint = vis.copy()
        for d in self.defects:
            tint[self.labels == d.id] = COLORS.get(d.cls, (255, 255, 255))
        vis = cv2.addWeighted(tint, 0.45, vis, 0.55, 0)
        for d in self.defects:
            x, y, w, h = d.bbox
            c = COLORS.get(d.cls, (255, 255, 255))
            cv2.rectangle(vis, (x - 3, y - 3), (x + w + 3, y + h + 3), c, 1)
            if draw_labels:
                txt = f"{d.cls} {d.length_mm:.2f}mm"
                cv2.putText(vis, txt, (x - 3, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
        banner = (0, 160, 0) if self.verdict == "OK" else (0, 0, 220)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), banner, -1)
        cv2.putText(vis, f"{self.verdict}  defects={len(self.defects)}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return vis


class Inspector:
    def __init__(self, cfg: Optional[InspectionConfig] = None,
                 classifier: Optional[LearnedClassifier] = None,
                 calibration: Optional[CameraCalibration] = None):
        self.cfg = cfg or InspectionConfig()
        self.classifier = classifier
        self.calibration = calibration

    # ------------------------------------------------------------------
    def inspect(self, image: np.ndarray, roi: Optional[np.ndarray] = None) -> InspectionResult:
        import time
        t0 = time.perf_counter()
        cfg = self.cfg
        if self.calibration is not None:
            image = self.calibration.undistort(image)
        gray = to_gray_f32(image)
        resid = flat_field(gray, cfg.flat_field_order)
        labels, score, maps = segment(resid, cfg, roi)

        px = cfg.pixel_size_mm
        defects: List[Defect] = []
        for i in range(1, int(labels.max()) + 1):
            mask = labels == i
            f = region_features(mask, maps["signed"], score)
            cls = self.classifier.predict(f) if self.classifier else rule_classify(f, cfg)
            ys, xs = np.nonzero(mask)
            d = Defect(
                id=i, cls=cls,
                bbox=(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
                centroid=(float(xs.mean()), float(ys.mean())),
                area_px=f["area_px"], area_mm2=f["area_px"] * px * px,
                length_mm=f["length_px"] * px, width_mm=f["width_px"] * px,
                contrast=f["contrast"], score=f["peak_score"],
            )
            limit = cfg.max_size_mm.get(cls)
            d.rejectable = limit is not None and d.length_mm > limit
            defects.append(d)

        reasons = [f"{d.cls} #{d.id}: {d.length_mm:.2f} mm > {cfg.max_size_mm[d.cls]:.2f} mm"
                   for d in defects if d.rejectable]
        if len(defects) > cfg.max_defect_count:
            reasons.append(f"defect count {len(defects)} > {cfg.max_defect_count}")
        verdict = "NG" if reasons else "OK"
        return InspectionResult(verdict, defects, labels, score, image,
                                (time.perf_counter() - t0) * 1000, reasons)

    def inspect_file(self, path: str | Path, roi: Optional[np.ndarray] = None) -> InspectionResult:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return self.inspect(img, roi)
