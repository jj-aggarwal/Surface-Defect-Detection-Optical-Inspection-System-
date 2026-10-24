import numpy as np
import pytest

from sdd import InspectionConfig, Inspector
from sdd.synthetic import generate_sample, DEFECT_CLASSES


@pytest.fixture(scope="module")
def inspector():
    return Inspector(InspectionConfig())


def test_clean_surfaces_pass(inspector):
    rng = np.random.default_rng(100)
    for _ in range(8):
        img, _ = generate_sample(rng, classes=[])
        res = inspector.inspect(img)
        assert res.verdict == "OK" and not res.defects


@pytest.mark.parametrize("cls", DEFECT_CLASSES)
def test_each_defect_type_is_found(inspector, cls):
    rng = np.random.default_rng(7)
    hits = 0
    for _ in range(6):
        img, gts = generate_sample(rng, classes=[cls])
        res = inspector.inspect(img)
        hits += bool(res.mask[gts[0].mask].any())
    assert hits >= 5, f"{cls}: detected {hits}/6"


def test_rule_classifier_reasonable(inspector):
    rng = np.random.default_rng(21)
    for cls in ("pit", "stain"):
        ok = 0
        for _ in range(8):
            img, gts = generate_sample(rng, classes=[cls])
            res = inspector.inspect(img)
            ok += any(d.cls == cls for d in res.defects)
        assert ok >= 7


def test_measurement_scales_with_pixel_size():
    rng = np.random.default_rng(3)
    img, _ = generate_sample(rng, classes=["scratch"])
    a = Inspector(InspectionConfig(pixel_size_mm=0.05)).inspect(img)
    b = Inspector(InspectionConfig(pixel_size_mm=0.10)).inspect(img)
    assert a.defects and b.defects
    assert b.defects[0].length_mm == pytest.approx(2 * a.defects[0].length_mm, rel=1e-6)


def test_crack_is_always_rejected():
    rng = np.random.default_rng(5)
    img, _ = generate_sample(rng, classes=["crack"])
    cfg = InspectionConfig()
    res = Inspector(cfg).inspect(img)
    if any(d.cls == "crack" for d in res.defects):
        assert res.verdict == "NG"


def test_roi_masks_out_defects():
    rng = np.random.default_rng(9)
    img, gts = generate_sample(rng, classes=["pit"])
    roi = np.ones(img.shape[:2], bool)
    ys, xs = np.nonzero(gts[0].mask)
    roi[max(ys.min() - 40, 0):ys.max() + 40, max(xs.min() - 40, 0):xs.max() + 40] = False
    assert not Inspector().inspect(img, roi=roi).defects


def test_overlay_and_dict(inspector):
    rng = np.random.default_rng(2)
    img, _ = generate_sample(rng, classes=["stain"])
    res = inspector.inspect(img)
    assert res.overlay().shape == img.shape
    d = res.to_dict()
    assert d["verdict"] in {"OK", "NG"} and d["n_defects"] == len(res.defects)


def test_config_yaml_roundtrip(tmp_path):
    cfg = InspectionConfig(pixel_size_mm=0.02)
    cfg.to_yaml(tmp_path / "c.yaml")
    assert InspectionConfig.from_yaml(tmp_path / "c.yaml").pixel_size_mm == 0.02
    (tmp_path / "bad.yaml").write_text("nonsense: 1")
    with pytest.raises(KeyError):
        InspectionConfig.from_yaml(tmp_path / "bad.yaml")
