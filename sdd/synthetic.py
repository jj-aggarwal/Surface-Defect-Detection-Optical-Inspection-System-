"""Synthetic brushed-metal surface generator with labelled defects.

Lets the whole project run and be benchmarked without a proprietary dataset.
Defect classes: scratch, crack, pit, stain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

DEFECT_CLASSES = ("scratch", "crack", "pit", "stain")


@dataclass
class GTDefect:
    cls: str
    mask: np.ndarray  # bool HxW

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        ys, xs = np.nonzero(self.mask)
        return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _texture(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    streak = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=6, sigmaY=0.8)
    streak /= streak.std() + 1e-6
    fine = rng.normal(0, 1, (h, w)).astype(np.float32)
    return 3.0 * streak + 2.0 * fine


def _illumination(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = xx / w - 0.5, yy / h - 0.5
    a, b = rng.uniform(-0.15, 0.15, 2)
    c = rng.uniform(-0.2, 0.2)
    return 1.0 + a * x + b * y + c * (x * x + y * y)


def _polyline(rng, h, w, n_pts, step, jitter, thickness, start=None, angle=None):
    margin = 25
    x = rng.uniform(margin, w - margin) if start is None else start[0]
    y = rng.uniform(margin, h - margin) if start is None else start[1]
    ang = rng.uniform(0, np.pi) if angle is None else angle
    pts = [(x, y)]
    for _ in range(n_pts):
        ang += rng.normal(0, jitter)
        x += step * np.cos(ang)
        y += step * np.sin(ang)
        if not (margin / 2 < x < w - margin / 2 and margin / 2 < y < h - margin / 2):
            break
        pts.append((x, y))
    m = np.zeros((h, w), np.uint8)
    cv2.polylines(m, [np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)], False, 255, thickness, cv2.LINE_AA)
    return m


def _make_defect(rng, cls: str, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (delta image to add, boolean GT mask)."""
    if cls == "scratch":
        m = _polyline(rng, h, w, n_pts=int(rng.integers(12, 30)), step=5, jitter=0.02,
                      thickness=int(rng.integers(1, 3)))
        amp = rng.choice([-1, 1]) * rng.uniform(35, 60)
        delta = amp * cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 0.7)
        return delta, m > 100
    if cls == "crack":
        m = _polyline(rng, h, w, n_pts=int(rng.integers(14, 30)), step=5, jitter=0.28,
                      thickness=2)
        delta = -rng.uniform(55, 90) * cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 0.7)
        return delta, m > 100
    if cls == "pit":
        m = np.zeros((h, w), np.uint8)
        c = (int(rng.integers(25, w - 25)), int(rng.integers(25, h - 25)))
        ax = (int(rng.integers(3, 8)), int(rng.integers(3, 8)))
        cv2.ellipse(m, c, ax, float(rng.uniform(0, 180)), 0, 360, 255, -1, cv2.LINE_AA)
        delta = -rng.uniform(55, 90) * cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 0.8)
        return delta, m > 100
    if cls == "stain":
        m = np.zeros((h, w), np.uint8)
        c = (int(rng.integers(50, w - 50)), int(rng.integers(50, h - 50)))
        ax = (int(rng.integers(18, 34)), int(rng.integers(14, 28)))
        cv2.ellipse(m, c, ax, float(rng.uniform(0, 180)), 0, 360, 255, -1)
        soft = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 5)
        delta = rng.choice([-1, 1]) * rng.uniform(10, 18) * soft
        return delta, soft > 0.5
    raise ValueError(cls)


def generate_sample(rng: np.random.Generator, size: Tuple[int, int] = (512, 512),
                    classes: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[GTDefect]]:
    """Generate one BGR uint8 image. Empty/None `classes` => defect-free part."""
    h, w = size
    img = 140.0 * _illumination(rng, h, w) + _texture(rng, h, w)
    gts: List[GTDefect] = []
    for cls in (classes or []):
        for _ in range(20):  # retry to avoid overlap with earlier defects
            delta, mask = _make_defect(rng, cls, h, w)
            if mask.sum() < 4:
                continue
            if any(cv2.dilate(g.mask.astype(np.uint8), np.ones((15, 15), np.uint8))[mask].any() for g in gts):
                continue
            img = img + delta
            gts.append(GTDefect(cls, mask))
            break
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gts


def random_classes(rng: np.random.Generator, max_defects: int = 3) -> List[str]:
    n = int(rng.integers(1, max_defects + 1))
    return [str(rng.choice(DEFECT_CLASSES)) for _ in range(n)]


def generate_dataset(out_dir: str | Path, n_good: int = 20, n_bad: int = 40, seed: int = 0,
                     size: Tuple[int, int] = (512, 512)) -> Path:
    """Write images/, masks/ (label map: 0=bg,1..4=class idx+1) and annotations.json."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    ann = []
    plan = [[] for _ in range(n_good)] + [random_classes(rng) for _ in range(n_bad)]
    for i, classes in enumerate(plan):
        img, gts = generate_sample(rng, size, classes)
        name = f"part_{i:04d}"
        cv2.imwrite(str(out / "images" / f"{name}.png"), img)
        label = np.zeros(size, np.uint8)
        for g in gts:
            label[g.mask] = DEFECT_CLASSES.index(g.cls) + 1
        cv2.imwrite(str(out / "masks" / f"{name}.png"), label)
        ann.append({"image": f"images/{name}.png", "mask": f"masks/{name}.png",
                    "label": "NG" if gts else "OK",
                    "defects": [{"class": g.cls, "bbox": list(g.bbox)} for g in gts]})
    (out / "annotations.json").write_text(json.dumps(ann, indent=2))
    return out
