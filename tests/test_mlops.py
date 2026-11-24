import json

import numpy as np
import pandas as pd

from mlops.evaluate import make_models, prepare_dataset, render_comparison, run_evaluation, score_model
from mlops.features import FEATURES, label_windows, window_features
from mlops.split import device_bucket, split_by_device
from mlops.tracking import Tracker


def test_window_features_shape(converted):
    feats = window_features(converted["df"])
    assert set(FEATURES) <= set(feats.columns)
    assert not feats[FEATURES].isna().any().any()
    assert len(feats) > 500


def test_windows_skip_gap_edges(converted):
    feats = window_features(converted["df"])
    assert feats["n"].min() >= 5


def test_labels_follow_anomaly_windows():
    feats = pd.DataFrame({"device_id": ["a", "a", "a", "b"], "window_start": [0.0, 10.0, 20.0, 10.0]})
    y = label_windows(feats, [{"device": "a", "start_ts": 8.0, "end_ts": 22.0}])
    assert y.tolist() == [0, 1, 0, 0]


def test_fault_windows_have_higher_vibration(converted, fleet):
    feats = window_features(converted["df"])
    y = label_windows(feats, fleet["gt"]["anomalies"])
    assert feats.loc[y == 1, "acc_rms"].mean() > 3 * feats.loc[y == 0, "acc_rms"].mean()


def test_split_is_grouped_and_deterministic(converted):
    feats = window_features(converted["df"])
    tr1, va1 = split_by_device(feats)
    tr2, va2 = split_by_device(feats)
    assert set(tr1.device_id) & set(va1.device_id) == set()
    assert set(va1.device_id) == set(va2.device_id)
    assert len(tr1) + len(va1) == len(feats)


def test_device_bucket_range():
    vals = [device_bucket(f"drone-{i}") for i in range(50)]
    assert min(vals) >= 0 and max(vals) <= 1 and len(set(vals)) == 50


def test_tracker_writes_mlflow_style_files(tmp_path):
    t = Tracker(tmp_path)
    run = t.start_run("exp", "m1", {"a": 1})
    run.log_metrics({"f1": 0.5})
    run.log_artifact_text("note.md", "hi")
    run.end()
    d = tmp_path / "exp" / run.run_id
    for f in ["meta.json", "params.json", "metrics.json", "tags.json", "artifacts/note.md"]:
        assert (d / f).exists(), f
    assert json.loads((d / "meta.json").read_text())["status"] == "FINISHED"


def test_tracker_best_run(tmp_path):
    t = Tracker(tmp_path)
    for name, f1 in [("a", 0.2), ("b", 0.9), ("c", 0.5)]:
        r = t.start_run("exp", name)
        r.log_metrics({"f1": f1})
        r.end()
    assert t.best_run("exp", "f1")["name"] == "b"
    assert len(t.list_runs("exp")) == 3


def test_run_id_deterministic(tmp_path):
    t = Tracker(tmp_path)
    assert t.start_run("e", "n", {"x": 1}).run_id == t.start_run("e", "n", {"x": 1}).run_id


def test_score_model_perfect_separation():
    rng = np.random.default_rng(0)
    x_train = pd.DataFrame(rng.normal(0, 1, (300, 2)), columns=["a", "b"])
    x_val = pd.DataFrame(np.vstack([rng.normal(0, 1, (90, 2)), rng.normal(12, 1, (10, 2))]), columns=["a", "b"])
    y = pd.Series([0] * 90 + [1] * 10)
    model = make_models()["isolation_forest"].fit(x_train)
    m = score_model(model, x_val, y)
    assert m["recall"] == 1.0 and m["roc_auc"] > 0.99


def test_evaluation_end_to_end(converted, fleet, tmp_path):
    summary = run_evaluation(converted["df"], fleet["gt"]["anomalies"], tmp_path / "mlruns", "v-test", tmp_path / "ds")
    assert set(summary["models"]) == {"isolation_forest", "local_outlier_factor"}
    assert (tmp_path / "ds" / "train.parquet").exists() and (tmp_path / "ds" / "val.parquet").exists()
    assert len(Tracker(tmp_path / "mlruns").list_runs("drone-anomaly")) == 2
    best = summary["models"][summary["best_model"]]
    assert best["roc_auc"] > 0.9  # injected motor faults must be separable
    assert "| isolation_forest |" in render_comparison(summary)


def test_evaluation_is_reproducible(converted, fleet, tmp_path):
    a = run_evaluation(converted["df"], fleet["gt"]["anomalies"], tmp_path / "r1", "v")
    b = run_evaluation(converted["df"], fleet["gt"]["anomalies"], tmp_path / "r2", "v")
    for name in a["models"]:
        assert a["models"][name]["f1"] == b["models"][name]["f1"]


def test_prepare_dataset_has_positive_labels_in_validation(converted, fleet):
    _, train, val = prepare_dataset(converted["df"], fleet["gt"]["anomalies"])
    assert val["label"].sum() > 0 and train["label"].sum() > 0
