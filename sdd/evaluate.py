"""Benchmark an Inspector on an annotated dataset (see `synthetic.generate_dataset`)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .pipeline import Inspector
from .synthetic import DEFECT_CLASSES


def _gt_instances(label_map: np.ndarray):
    for ci, cls in enumerate(DEFECT_CLASSES, start=1):
        n, lab = cv2.connectedComponents((label_map == ci).astype(np.uint8), connectivity=8)
        for j in range(1, n):
            yield cls, lab == j


def evaluate(inspector: Inspector, dataset_dir: str | Path, min_gt_overlap: float = 0.3) -> Dict:
    root = Path(dataset_dir)
    ann = json.loads((root / "annotations.json").read_text())
    tp = fp = fn = 0                       # defect level
    img_tp = img_fp = img_fn = img_tn = 0  # image level (any defect detected vs GT NG)
    verdict_ok = 0
    cls_conf = {a: {b: 0 for b in DEFECT_CLASSES} for a in DEFECT_CLASSES}
    inter_px = union_px = 0
    times: List[float] = []

    for item in ann:
        img = cv2.imread(str(root / item["image"]), cv2.IMREAD_COLOR)
        gt_map = cv2.imread(str(root / item["mask"]), cv2.IMREAD_GRAYSCALE)
        res = inspector.inspect(img)
        times.append(res.elapsed_ms)
        gts = list(_gt_instances(gt_map))
        has_gt, has_det = bool(gts), bool(res.defects)
        img_tp += has_gt and has_det
        img_fn += has_gt and not has_det
        img_fp += (not has_gt) and has_det
        img_tn += (not has_gt) and (not has_det)

        matched_det = set()
        for cls, gmask in gts:
            best, best_ov = None, 0.0
            for d in res.defects:
                ov = float((res.labels == d.id)[gmask].sum()) / gmask.sum()
                if ov > best_ov:
                    best, best_ov = d, ov
            if best is not None and best_ov >= min_gt_overlap:
                tp += 1
                matched_det.add(best.id)
                cls_conf[cls][best.cls] += 1
            else:
                fn += 1
        fp += sum(1 for d in res.defects if d.id not in matched_det and
                  not any((res.labels == d.id)[g].any() for _, g in gts))

        gt_union = gt_map > 0
        inter_px += int((res.mask & gt_union).sum())
        union_px += int((res.mask | gt_union).sum())

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    cls_total = sum(sum(r.values()) for r in cls_conf.values())
    cls_correct = sum(cls_conf[c][c] for c in DEFECT_CLASSES)
    per_class_recall = {c: cls_conf[c][c] / max(sum(cls_conf[c].values()), 1) for c in DEFECT_CLASSES}
    return {
        "n_images": len(ann),
        "defect_precision": round(prec, 4), "defect_recall": round(rec, 4),
        "defect_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "pixel_iou": round(inter_px / max(union_px, 1), 4),
        "image_level": {"tp": img_tp, "fp": img_fp, "fn": img_fn, "tn": img_tn,
                        "accuracy": round((img_tp + img_tn) / max(len(ann), 1), 4)},
        "classification_accuracy": round(cls_correct / max(cls_total, 1), 4),
        "per_class_recall": {k: round(v, 4) for k, v in per_class_recall.items()},
        "confusion_matrix": cls_conf,
        "mean_latency_ms": round(float(np.mean(times)), 1),
        "p95_latency_ms": round(float(np.percentile(times, 95)), 1),
    }


def to_markdown(m: Dict) -> str:
    cm = m["confusion_matrix"]
    rows = "\n".join(f"| {a} | " + " | ".join(str(cm[a][b]) for b in DEFECT_CLASSES) + " |" for a in DEFECT_CLASSES)
    il = m["image_level"]
    return f"""# Evaluation report

| Metric | Value |
|---|---|
| Images | {m['n_images']} |
| Defect precision | {m['defect_precision']:.3f} |
| Defect recall | {m['defect_recall']:.3f} |
| Defect F1 | {m['defect_f1']:.3f} |
| Pixel IoU | {m['pixel_iou']:.3f} |
| Classification accuracy (matched defects) | {m['classification_accuracy']:.3f} |
| Image-level accuracy | {il['accuracy']:.3f} (TP {il['tp']} / FP {il['fp']} / FN {il['fn']} / TN {il['tn']}) |
| Latency mean / p95 | {m['mean_latency_ms']} ms / {m['p95_latency_ms']} ms |

**Confusion matrix** (rows = ground truth, columns = predicted)

| | {' | '.join(DEFECT_CLASSES)} |
|---|{'---|' * len(DEFECT_CLASSES)}
{rows}
"""
