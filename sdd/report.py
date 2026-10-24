"""Batch reporting: JSON, CSV and a self-contained HTML report with thumbnails."""
from __future__ import annotations

import base64
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2


def write_json(rows: List[Dict], path: Path) -> None:
    path.write_text(json.dumps(rows, indent=2))


def write_csv(rows: List[Dict], path: Path) -> None:
    cols = ["image", "verdict", "defect_id", "class", "length_mm", "width_mm", "area_mm2",
            "contrast", "score", "rejectable", "x", "y", "w", "h"]
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for r in rows:
            if not r["defects"]:
                wr.writerow([r["image"], r["verdict"]] + [""] * (len(cols) - 2))
            for d in r["defects"]:
                x, y, w, h = d["bbox"]
                wr.writerow([r["image"], r["verdict"], d["id"], d["cls"], f"{d['length_mm']:.3f}",
                             f"{d['width_mm']:.3f}", f"{d['area_mm2']:.4f}", f"{d['contrast']:.1f}",
                             f"{d['score']:.1f}", d["rejectable"], x, y, w, h])


def _b64(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


def write_html(rows: List[Dict], overlays: Dict[str, "np.ndarray"], path: Path, title="Inspection report") -> None:
    n_ng = sum(r["verdict"] == "NG" for r in rows)
    cards = []
    for r in rows:
        thumb = cv2.resize(overlays[r["image"]], None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        items = "".join(f"<li>{html.escape(d['cls'])} — {d['length_mm']:.2f} mm"
                        f"{' ⚠' if d['rejectable'] else ''}</li>" for d in r["defects"]) or "<li>no defects</li>"
        cards.append(f"<div class='card {r['verdict']}'><img src='data:image/jpeg;base64,{_b64(thumb)}'>"
                     f"<b>{html.escape(r['image'])}</b> <span class='v'>{r['verdict']}</span><ul>{items}</ul></div>")
    path.write_text(f"""<!doctype html><meta charset='utf-8'><title>{title}</title>
<style>body{{font-family:system-ui;margin:24px;background:#111;color:#eee}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}}
.card{{background:#1c1c1c;padding:10px;border-radius:8px;border-left:6px solid #2a2}}
.card.NG{{border-left-color:#d22}} img{{width:100%;border-radius:4px}} .v{{float:right;font-weight:700}}
ul{{margin:6px 0 0 18px;padding:0;font-size:13px}}</style>
<h1>{title}</h1><p>{datetime.now():%Y-%m-%d %H:%M} — {len(rows)} parts, {n_ng} NG, {len(rows)-n_ng} OK
(yield {100*(len(rows)-n_ng)/max(len(rows),1):.1f}%)</p><div class='grid'>{''.join(cards)}</div>""")
