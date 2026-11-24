"""Deterministic train/validation split grouped by device.

Splitting by device (not by row) keeps windows from the same flight out of
both sets, so validation measures how a model does on an unseen drone.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def device_bucket(device: str, salt: str = "split-v1") -> float:
    h = hashlib.sha256(f"{salt}:{device}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_by_device(
    feats: pd.DataFrame, val_fraction: float = 0.34, salt: str = "split-v1"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    devices = sorted(feats["device_id"].unique())
    ranked = sorted(devices, key=lambda d: device_bucket(d, salt))
    n_val = max(1, round(len(devices) * val_fraction))
    if n_val >= len(devices):
        n_val = len(devices) - 1
    val_devs = set(ranked[:n_val])
    val = feats[feats["device_id"].isin(val_devs)].reset_index(drop=True)
    train = feats[~feats["device_id"].isin(val_devs)].reset_index(drop=True)
    return train, val
