"""Pre-processing: grayscale conversion and illumination flat-fielding."""
from __future__ import annotations

import cv2
import numpy as np


def to_gray_f32(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def _design(xn: np.ndarray, yn: np.ndarray, order: int) -> np.ndarray:
    cols = [xn ** i * yn ** j for i in range(order + 1) for j in range(order + 1 - i)]
    return np.stack(cols, axis=-1)


def flat_field(gray: np.ndarray, order: int = 2, iters: int = 4, work: int = 96) -> np.ndarray:
    """Subtract a robustly fitted polynomial illumination surface.

    Iterative sigma-clipping keeps defects (outliers) from biasing the fit, unlike a
    large Gaussian blur which would partially absorb big stains.
    """
    h, w = gray.shape
    small = cv2.resize(gray, (work, work), interpolation=cv2.INTER_AREA)
    ys, xs = np.mgrid[0:work, 0:work]
    A = _design((xs / (work - 1) * 2 - 1).ravel(), (ys / (work - 1) * 2 - 1).ravel(), order)
    z = small.ravel()
    keep = np.ones_like(z, bool)
    coef = None
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        res = z - A @ coef
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826 + 1e-6
        keep = np.abs(res) < 2.5 * mad
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    Af = _design(xx / (w - 1) * 2 - 1, yy / (h - 1) * 2 - 1, order)
    bg = Af @ coef
    return gray - bg.astype(np.float32)
