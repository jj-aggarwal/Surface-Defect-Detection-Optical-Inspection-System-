"""Command line interface:  python -m sdd.cli <generate|inspect|evaluate|train|calibrate>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .classify import FEATURE_NAMES, LearnedClassifier, region_features
from .config import InspectionConfig
from .evaluate import evaluate, to_markdown
from .optics import CameraCalibration, calibrate_camera
from .pipeline import Inspector
from .report import write_csv, write_html, write_json
from .synthetic import DEFECT_CLASSES, generate_dataset

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _inspector(a) -> Inspector:
    cfg = InspectionConfig.from_yaml(a.config) if getattr(a, "config", None) else InspectionConfig()
    if getattr(a, "pixel_size", None):
        cfg.pixel_size_mm = a.pixel_size
    clf = LearnedClassifier.load(a.model) if getattr(a, "model", None) else None
    cal = CameraCalibration.load(a.calibration) if getattr(a, "calibration", None) else None
    return Inspector(cfg, clf, cal)


def cmd_generate(a):
    out = generate_dataset(a.out, a.n_good, a.n_bad, a.seed)
    print(f"dataset written to {out}")


def cmd_inspect(a):
    ins = _inspector(a)
    src = Path(a.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, overlays = [], {}
    for f in files:
        res = ins.inspect_file(f)
        ov = res.overlay()
        cv2.imwrite(str(out / f"{f.stem}_overlay.png"), ov)
        d = res.to_dict()
        d["image"] = f.name
        rows.append(d)
        overlays[f.name] = ov
        print(f"{f.name:28s} {res.verdict}  defects={len(res.defects)}  {res.elapsed_ms:.0f} ms")
    write_json(rows, out / "results.json")
    write_csv(rows, out / "results.csv")
    write_html(rows, overlays, out / "report.html")
    ng = sum(r["verdict"] == "NG" for r in rows)
    print(f"\n{len(rows)} parts: {ng} NG / {len(rows) - ng} OK -> {out}/report.html")


def cmd_evaluate(a):
    m = evaluate(_inspector(a), a.data)
    print(json.dumps({k: v for k, v in m.items() if k != "confusion_matrix"}, indent=2))
    if a.out:
        Path(a.out).write_text(to_markdown(m))
        print(f"markdown report -> {a.out}")


def cmd_train(a):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from .detect import segment
    from .preprocess import flat_field, to_gray_f32
    cfg = InspectionConfig()
    root = Path(a.data)
    ann = json.loads((root / "annotations.json").read_text())
    X, y = [], []
    for item in ann:
        img = cv2.imread(str(root / item["image"]))
        gt = cv2.imread(str(root / item["mask"]), cv2.IMREAD_GRAYSCALE)
        labels, score, maps = segment(flat_field(to_gray_f32(img), cfg.flat_field_order), cfg)
        for i in range(1, int(labels.max()) + 1):
            m = labels == i
            vals = gt[m][gt[m] > 0]
            if vals.size == 0:
                continue
            y.append(DEFECT_CLASSES[int(np.bincount(vals).argmax()) - 1])
            f = region_features(m, maps["signed"], score)
            X.append([f[k] for k in FEATURE_NAMES])
    X, y = np.array(X, np.float32), np.array(y)
    clf = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced")
    print(f"{len(y)} labelled regions; 5-fold CV accuracy = {cross_val_score(clf, X, y, cv=5).mean():.3f}")
    clf.fit(X, y)
    LearnedClassifier(clf).save(a.out)
    print(f"model saved -> {a.out}")


def cmd_calibrate(a):
    imgs = [cv2.imread(str(p)) for p in sorted(Path(a.images).glob("*")) if p.suffix.lower() in IMG_EXT]
    cols, rows = map(int, a.pattern.split("x"))
    cal = calibrate_camera([i for i in imgs if i is not None], (cols, rows), a.square_mm)
    cal.save(a.out)
    print(f"RMS reprojection error: {cal.rms_error_px:.3f} px | approx scale: {cal.pixel_size_mm:.4f} mm/px -> {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sdd", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create a synthetic labelled dataset")
    g.add_argument("--out", default="data/synthetic")
    g.add_argument("--n-good", type=int, default=20)
    g.add_argument("--n-bad", type=int, default=60)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_generate)

    def common(sp):
        sp.add_argument("--config")
        sp.add_argument("--pixel-size", type=float, help="mm per pixel (overrides config)")
        sp.add_argument("--model", help="joblib classifier from `train`")
        sp.add_argument("--calibration", help="calibration JSON from `calibrate`")

    i = sub.add_parser("inspect", help="inspect an image or a folder of images")
    i.add_argument("input")
    i.add_argument("--out", default="outputs")
    common(i)
    i.set_defaults(fn=cmd_inspect)

    e = sub.add_parser("evaluate", help="benchmark on an annotated dataset")
    e.add_argument("data")
    e.add_argument("--out", help="write markdown report here")
    common(e)
    e.set_defaults(fn=cmd_evaluate)

    t = sub.add_parser("train", help="train the optional ML defect classifier")
    t.add_argument("data")
    t.add_argument("--out", default="models/classifier.joblib")
    t.set_defaults(fn=cmd_train)

    c = sub.add_parser("calibrate", help="camera calibration from checkerboard images")
    c.add_argument("images")
    c.add_argument("--pattern", default="9x6", help="inner corners, COLSxROWS")
    c.add_argument("--square-mm", type=float, default=5.0)
    c.add_argument("--out", default="calibration.json")
    c.set_defaults(fn=cmd_calibrate)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
