"""Interactive inspection station UI:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdd import InspectionConfig, Inspector  # noqa: E402
from sdd.synthetic import DEFECT_CLASSES, generate_sample  # noqa: E402

st.set_page_config(page_title="Surface Defect Inspection", layout="wide")
st.title("Surface Defect Detection & Optical Inspection")

with st.sidebar:
    st.header("Optics")
    px = st.number_input("Pixel size (mm/px)", 0.001, 1.0, 0.05, 0.005, format="%.3f")
    st.header("Detection")
    high = st.slider("Seed threshold (robust z)", 4.0, 12.0, 6.5, 0.5)
    low = st.slider("Region-growing threshold", 2.5, 6.0, 4.0, 0.5)
    min_area = st.slider("Min area (px)", 4, 100, 10)
    st.header("Accept / reject (max length mm)")
    limits = {c: st.number_input(c, 0.0, 20.0, v, 0.1)
              for c, v in {"scratch": 3.0, "crack": 0.0, "pit": 0.5, "stain": 2.0}.items()}

src = st.radio("Image source", ["Upload", "Synthetic demo"], horizontal=True)
img = None
if src == "Upload":
    up = st.file_uploader("Surface image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
else:
    c1, c2 = st.columns(2)
    picks = c1.multiselect("Defects to inject", DEFECT_CLASSES, default=["scratch", "pit"])
    seed = c2.number_input("Seed", 0, 10_000, 0)
    img, _ = generate_sample(np.random.default_rng(int(seed)), classes=picks)

if img is not None:
    cfg = InspectionConfig(pixel_size_mm=px, high_threshold=high, low_threshold=low,
                           min_area_px=min_area, max_size_mm=limits)
    res = Inspector(cfg).inspect(img)
    left, right = st.columns(2)
    left.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
    right.image(cv2.cvtColor(res.overlay(), cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)
    (st.success if res.verdict == "OK" else st.error)(f"Verdict: {res.verdict}  ({res.elapsed_ms:.0f} ms)")
    for r in res.reasons:
        st.warning(r)
    if res.defects:
        st.dataframe([{k: v for k, v in d.__dict__.items() if k not in ("centroid",)} for d in res.defects])
