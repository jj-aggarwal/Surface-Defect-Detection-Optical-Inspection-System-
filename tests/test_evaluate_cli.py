import json

from sdd.cli import main
from sdd.config import InspectionConfig
from sdd.evaluate import evaluate
from sdd.pipeline import Inspector
from sdd.synthetic import generate_dataset


def test_benchmark_meets_targets(tmp_path):
    generate_dataset(tmp_path, n_good=8, n_bad=24, seed=42)
    m = evaluate(Inspector(InspectionConfig()), tmp_path)
    assert m["defect_recall"] >= 0.9
    assert m["defect_precision"] >= 0.9
    assert m["image_level"]["fp"] == 0
    assert m["classification_accuracy"] >= 0.75


def test_cli_end_to_end(tmp_path, capsys):
    data = tmp_path / "ds"
    main(["generate", "--out", str(data), "--n-good", "2", "--n-bad", "4", "--seed", "1"])
    out = tmp_path / "out"
    main(["inspect", str(data / "images"), "--out", str(out)])
    assert (out / "report.html").exists() and (out / "results.csv").exists()
    rows = json.loads((out / "results.json").read_text())
    assert len(rows) == 6


def test_train_and_use_model(tmp_path):
    data = tmp_path / "ds"
    generate_dataset(data, n_good=0, n_bad=30, seed=5)
    model = tmp_path / "m.joblib"
    main(["train", str(data), "--out", str(model)])
    from sdd.classify import LearnedClassifier
    ins = Inspector(InspectionConfig(), classifier=LearnedClassifier.load(model))
    import cv2
    res = ins.inspect(cv2.imread(str(data / "images" / "part_0000.png")))
    assert res.verdict in {"OK", "NG"}
