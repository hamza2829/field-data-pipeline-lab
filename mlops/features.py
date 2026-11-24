"""Window features for anomaly detection plus ground truth labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["acc_rms", "acc_std", "gyro_rms", "battery_v_mean", "battery_v_slope", "temp_mean", "temp_delta"]


def window_features(curated: pd.DataFrame, window_s: int = 10) -> pd.DataFrame:
    df = curated.sort_values(["device_id", "ts"]).copy()
    df["acc_mag"] = np.sqrt(df.acc_x**2 + df.acc_y**2 + (df.acc_z - 9.81) ** 2)
    df["gyro_mag"] = np.sqrt(df.gyro_x**2 + df.gyro_y**2 + df.gyro_z**2)
    df["window_start"] = (df["ts"] // window_s) * window_s
    rows = []
    for (dev, w0), g in df.groupby(["device_id", "window_start"], sort=True):
        if len(g) < max(3, window_s // 2):
            continue  # too few samples (gap edge)
        t = g["ts"].to_numpy() - w0
        bv = g["battery_v"].to_numpy()
        slope = float(np.polyfit(t, bv, 1)[0]) if len(g) > 1 else 0.0
        rows.append(
            {
                "device_id": str(dev),
                "window_start": float(w0),
                "n": len(g),
                "acc_rms": float(np.sqrt(np.mean(g.acc_mag**2))),
                "acc_std": float(g.acc_mag.std(ddof=0)),
                "gyro_rms": float(np.sqrt(np.mean(g.gyro_mag**2))),
                "battery_v_mean": float(bv.mean()),
                "battery_v_slope": slope,
                "temp_mean": float(g.temp_c.mean()),
                "temp_delta": 0.0,
            }
        )
    feats = pd.DataFrame(rows)
    # temperature relative to the device's own median removes the fleet offset
    feats["temp_delta"] = feats["temp_mean"] - feats.groupby("device_id")["temp_mean"].transform("median")
    return feats


def label_windows(feats: pd.DataFrame, anomalies: list[dict], window_s: int = 10) -> pd.Series:
    """A window is anomalous when at least half of it lies inside an injected fault."""
    labels = np.zeros(len(feats), dtype=int)
    by_dev: dict[str, list[tuple[float, float]]] = {}
    for a in anomalies:
        by_dev.setdefault(a["device"], []).append((a["start_ts"], a["end_ts"]))
    for i, (dev, w0) in enumerate(zip(feats["device_id"], feats["window_start"])):
        w1 = w0 + window_s
        overlap = sum(max(0.0, min(w1, e) - max(w0, s)) for s, e in by_dev.get(dev, []))
        labels[i] = int(overlap >= window_s / 2)
    return pd.Series(labels, index=feats.index, name="label")
