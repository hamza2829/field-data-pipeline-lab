"""Automated evaluation of two unsupervised anomaly models.

Both models are trained only on the training devices, without labels.
The injected fault windows from the simulator are used only for scoring on
the validation devices.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlops.features import FEATURES, label_windows, window_features
from mlops.split import split_by_device
from mlops.tracking import Tracker

CONTAMINATION = 0.1


def make_models(seed: int = 7) -> dict:
    return {
        "isolation_forest": make_pipeline(
            StandardScaler(), IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=seed)
        ),
        "local_outlier_factor": make_pipeline(
            StandardScaler(), LocalOutlierFactor(n_neighbors=25, contamination=CONTAMINATION, novelty=True)
        ),
    }


def score_model(model, x_val: pd.DataFrame, y_val: pd.Series) -> dict:
    pred = (model.predict(x_val) == -1).astype(int)
    score = -model.score_samples(x_val)  # higher = more anomalous
    return {
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, score) if y_val.nunique() > 1 else float("nan"),
        "avg_precision": average_precision_score(y_val, score) if y_val.nunique() > 1 else float("nan"),
    }


def prepare_dataset(curated: pd.DataFrame, anomalies: list[dict], window_s: int = 10):
    feats = window_features(curated, window_s)
    feats["label"] = label_windows(feats, anomalies, window_s)
    train, val = split_by_device(feats)
    return feats, train, val


def run_evaluation(
    curated: pd.DataFrame,
    anomalies: list[dict],
    mlruns_dir: str | Path,
    dataset_version: str,
    out_dir: str | Path | None = None,
    seed: int = 7,
) -> dict:
    feats, train, val = prepare_dataset(curated, anomalies)
    if out_dir:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        train.to_parquet(od / "train.parquet", index=False)
        val.to_parquet(od / "val.parquet", index=False)

    tracker = Tracker(mlruns_dir)
    results = {}
    for name, model in make_models(seed).items():
        params = {
            "model": name,
            "features": FEATURES,
            "contamination": CONTAMINATION,
            "seed": seed,
            "dataset_version": dataset_version,
        }
        run = tracker.start_run("drone-anomaly", name, params)
        run.set_tags(
            {
                "train_devices": sorted(train.device_id.unique().tolist()),
                "val_devices": sorted(val.device_id.unique().tolist()),
            }
        )
        model.fit(train[FEATURES])
        metrics = score_model(model, val[FEATURES], val["label"])
        run.log_metrics(metrics)
        run.end()
        results[name] = {"run_id": run.run_id, **metrics}

    best = max(results, key=lambda k: results[k]["f1"])
    summary = {
        "dataset_version": dataset_version,
        "windows": int(len(feats)),
        "train_windows": int(len(train)),
        "val_windows": int(len(val)),
        "val_positive": int(val["label"].sum()),
        "train_devices": sorted(train.device_id.unique().tolist()),
        "val_devices": sorted(val.device_id.unique().tolist()),
        "models": results,
        "best_model": best,
    }
    return summary


def render_comparison(summary: dict) -> str:
    lines = [
        "# Model comparison",
        "",
        f"Dataset version `{summary['dataset_version']}`, {summary['windows']} windows of 10 s "
        f"({summary['train_windows']} train on {', '.join(summary['train_devices'])}; "
        f"{summary['val_windows']} validation on {', '.join(summary['val_devices'])}, "
        f"{summary['val_positive']} of them labelled as fault).",
        "",
        "| Model | Precision | Recall | F1 | ROC AUC | Avg precision | Run |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, m in summary["models"].items():
        lines.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} | "
            f"{m['avg_precision']:.3f} | `{m['run_id']}` |"
        )
    lines += ["", f"Best by F1: **{summary['best_model']}**", ""]
    return "\n".join(lines)
