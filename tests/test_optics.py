import cv2
import numpy as np
import pytest

from sdd.optics import calibrate_camera, make_checkerboard, pixel_size_from_reference


F_TRUE = 900.0


def _views(n=14, seed=0):
    """Render a checkerboard from n physically consistent camera poses (known pinhole K)."""
    board = make_checkerboard((9, 6), 40, 60)
    bh, bw = board.shape
    W, H = 640, 480
    K = np.array([[F_TRUE, 0, W / 2], [0, F_TRUE, H / 2], [0, 0, 1]])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        rvec = rng.uniform(-0.6, 0.6, 3) * [1, 1, 0.4]
        R, _ = cv2.Rodrigues(rvec)
        t = np.array([rng.uniform(-30, 30), rng.uniform(-20, 20), rng.uniform(600, 760)])
        # board pixel (u,v) -> metric plane coords (mm) centred on the board, 1 board px = 0.5 mm
        # (so a 40 px square is 20 mm; rendering stays ~1:1 and free of aliasing)
        S = np.array([[0.5, 0, -bw * 0.5 / 2], [0, 0.5, -bh * 0.5 / 2], [0, 0, 1]])
        H_ = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
        out.append(cv2.warpPerspective(board, H_, (W, H), borderValue=255))
    return out


def test_pixel_size_from_reference():
    assert pixel_size_from_reference(200, 10.0) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        pixel_size_from_reference(0, 1)


def test_calibration_recovers_scale_and_low_error():
    cal = calibrate_camera(_views(), (9, 6), square_mm=20.0)
    assert cal.rms_error_px < 1.0
    assert cal.camera_matrix[0, 0] == pytest.approx(F_TRUE, rel=0.05)
    assert 0.5 < cal.pixel_size_mm < 1.3     # ~ z / f = 0.67-0.84 mm per pixel


def test_calibration_needs_enough_views():
    with pytest.raises(RuntimeError):
        calibrate_camera([np.zeros((100, 100), np.uint8)] * 2, (9, 6))


def test_calibration_roundtrip(tmp_path):
    from sdd.optics import CameraCalibration
    cal = calibrate_camera(_views(), (9, 6), 20.0)
    cal.save(tmp_path / "c.json")
    back = CameraCalibration.load(tmp_path / "c.json")
    assert np.allclose(back.camera_matrix, cal.camera_matrix)
    img = _views(1)[0]
    assert back.undistort(img).shape == img.shape
