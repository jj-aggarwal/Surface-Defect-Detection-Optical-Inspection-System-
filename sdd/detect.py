"""Unsupervised defect segmentation (no training data required).

Three complementary robust-z-score maps are fused:
  * fine   – Gaussian-smoothed residual   -> pits, dents, small blobs
  * coarse – heavily smoothed residual    -> faint large stains / discoloration
  * ridge  – Hessian line detector        -> scratches & cracks (bright or dark)
Hysteresis thresholding (low/high) keeps weak pixels only if connected to a strong seed.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import InspectionConfig


def robust_z(m: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    v = m[valid] if valid is not None else m.ravel()
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    return (m - med) / mad


def ridge_response(g: np.ndarray, sigmas) -> np.ndarray:
    best = np.zeros_like(g)
    for s in sigmas:
        b = cv2.GaussianBlur(g, (0, 0), s)
        ixx = cv2.Sobel(b, cv2.CV_32F, 2, 0, ksize=3)
        iyy = cv2.Sobel(b, cv2.CV_32F, 0, 2, ksize=3)
        ixy = cv2.Sobel(b, cv2.CV_32F, 1, 1, ksize=3)
        mean = (ixx + iyy) / 2
        root = np.sqrt(((ixx - iyy) / 2) ** 2 + ixy ** 2)
        l1, l2 = mean + root, mean - root
        big = np.where(np.abs(l1) > np.abs(l2), l1, l2)
        small = np.where(np.abs(l1) > np.abs(l2), l2, l1)
        line = np.maximum(np.abs(big) - np.abs(small), 0) * s * s   # elongated, not blob
        best = np.maximum(best, line)
    return best


def anomaly_maps(resid: np.ndarray, cfg: InspectionConfig, valid: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    fine = cv2.GaussianBlur(resid, (0, 0), cfg.fine_sigma)
    coarse = cv2.GaussianBlur(resid, (0, 0), cfg.coarse_sigma)
    return {
        "fine": robust_z(fine, valid),
        "coarse": robust_z(coarse, valid),
        "ridge": robust_z(ridge_response(resid, cfg.ridge_sigmas), valid),
        "signed": fine,
    }


def segment(resid: np.ndarray, cfg: InspectionConfig, roi: Optional[np.ndarray] = None
            ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Return (label image, score map, maps). Labels: 0=bg, 1..N components."""
    h, w = resid.shape
    valid = np.ones((h, w), bool) if roi is None else roi.astype(bool)
    m = cfg.border_margin_px
    if m > 0:
        valid[:m, :] = valid[-m:, :] = False
        valid[:, :m] = valid[:, -m:] = False

    maps = anomaly_maps(resid, cfg, valid)
    sharp = np.maximum(np.abs(maps["fine"]), cfg.ridge_weight * maps["ridge"])   # pits, scratches, cracks
    soft = np.abs(maps["coarse"])                                                # stains
    sharp = np.where(valid, sharp, 0).astype(np.float32)
    soft = np.where(valid, soft, 0).astype(np.float32)
    score = np.maximum(sharp, soft)

    low = (score >= cfg.low_threshold).astype(np.uint8)
    if cfg.morph_close_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_close_px, cfg.morph_close_px))
        low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)

    k3 = np.ones((3, 3), np.uint8)
    out = np.zeros_like(lab, np.int32)
    k_out = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < cfg.min_area_px:
            continue
        comp = lab == i
        if score[comp].max() < cfg.high_threshold:
            continue
        # Scale selection: a region dominated by the coarse scale is a soft stain and is kept
        # whole; otherwise tighten it to the sharp (fine/ridge) response so that cracks, pits
        # and scratches keep their true shape instead of the blurred coarse footprint.
        is_stain = soft[comp].max() > 1.1 * sharp[comp].max() and sharp[comp].max() < cfg.stain_max_sharp
        if not is_stain:
            cut = max(cfg.low_threshold, cfg.peak_fraction * sharp[comp].max())
            comp = cv2.morphologyEx((comp & (sharp >= cut)).astype(np.uint8),
                                    cv2.MORPH_CLOSE, k3).astype(bool) & comp
            if comp.sum() < cfg.min_area_px:
                continue
        k_out += 1
        out[comp] = k_out
    return out, score, maps
