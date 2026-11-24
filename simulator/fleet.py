"""Seeded drone fleet simulator.

Every device produces three kinds of raw field data:

* ``sensors.csv``  - 1 Hz IMU, GPS, battery and temperature samples
* ``flight.log``   - free text log lines as an onboard computer would write them
* ``frames/``      - tiny greyscale camera frames (binary PGM) plus ``frames.json`` metadata

Defects are injected on purpose so the ingest and DataOps layers have
something to find. Everything injected is written to ``ground_truth.json`` so
tests can check that every defect is detected.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

START_TS = 1_795_500_000  # fixed epoch start, keeps every run reproducible
SENSOR_COLUMNS = [
    "ts",
    "device_id",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "lat",
    "lon",
    "alt_m",
    "battery_v",
    "battery_pct",
    "temp_c",
]
FRAME_SIZE = 32
LOG_COMPONENTS = ["nav", "motor", "power", "camera", "link"]


@dataclass
class FleetConfig:
    n_devices: int = 6
    seconds: int = 900
    seed: int = 42
    frame_every_s: int = 30
    defects: bool = True
    anomalies_per_device: int = 3
    anomaly_len_s: int = 40


@dataclass
class GroundTruth:
    gaps: list[dict] = field(default_factory=list)
    duplicates: list[dict] = field(default_factory=list)
    wrong_units: list[dict] = field(default_factory=list)
    corrupt_files: list[dict] = field(default_factory=list)
    bad_log_lines: list[dict] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gaps": self.gaps,
            "duplicates": self.duplicates,
            "wrong_units": self.wrong_units,
            "corrupt_files": self.corrupt_files,
            "bad_log_lines": self.bad_log_lines,
            "anomalies": self.anomalies,
        }


def device_ids(n: int) -> list[str]:
    return [f"drone-{i:02d}" for i in range(1, n + 1)]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pgm_bytes(img: np.ndarray) -> bytes:
    h, w = img.shape
    return f"P5\n{w} {h}\n255\n".encode() + img.astype(np.uint8).tobytes()


def _sensor_rows(dev_idx: int, cfg: FleetConfig, rng: np.random.Generator):
    n = cfg.seconds
    t = np.arange(n, dtype=float)
    ts = START_TS + t
    # anomaly windows (motor fault): high vibration + battery sag + heat
    anomaly_mask = np.zeros(n, dtype=bool)
    windows = []
    if cfg.anomalies_per_device:
        slots = np.linspace(60, n - cfg.anomaly_len_s - 60, cfg.anomalies_per_device)
        for s in slots:
            start = int(s + rng.integers(-20, 20))
            end = start + cfg.anomaly_len_s
            anomaly_mask[start:end] = True
            windows.append((start, end))
    vib = np.where(anomaly_mask, 2.2, 0.15)
    acc = rng.normal(0, 1, (n, 3)) * vib[:, None]
    acc[:, 2] += 9.81
    gyro = rng.normal(0, 0.02, (n, 3)) + np.where(anomaly_mask, 0.6, 0.0)[:, None] * rng.normal(0, 1, (n, 3))
    base_lat, base_lon = 48.137 + dev_idx * 0.01, 11.575 + dev_idx * 0.01
    radius = 0.002
    lat = base_lat + radius * np.sin(2 * math.pi * t / 300)
    lon = base_lon + radius * np.cos(2 * math.pi * t / 300)
    alt = 60 + 10 * np.sin(2 * math.pi * t / 120) + rng.normal(0, 0.3, n)
    battery_pct = np.clip(100 - t * (60 / n) - np.cumsum(anomaly_mask) * 0.05, 0, 100)
    battery_v = 12.0 + 4.8 * battery_pct / 100 + rng.normal(0, 0.02, n) - np.where(anomaly_mask, 0.6, 0.0)
    temp = 31 + 0.004 * t + rng.normal(0, 0.3, n) + np.where(anomaly_mask, 14.0, 0.0)
    rows = np.column_stack([ts, acc, gyro, lat, lon, alt, battery_v, battery_pct, temp])
    return rows, windows


def _write_sensors(path: Path, device: str, rows: list[list]) -> None:
    lines = [",".join(SENSOR_COLUMNS)]
    for r in rows:
        vals = [f"{r[0]:.1f}", device] + [f"{v:.6f}" for v in r[1:]]
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n")


def simulate_fleet(out_dir: str | Path, cfg: FleetConfig | None = None) -> dict:
    """Generate raw field data for the whole fleet. Returns a summary dict."""
    cfg = cfg or FleetConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    gt = GroundTruth()
    summary = {"devices": [], "sensor_rows": 0, "log_lines": 0, "frames": 0}

    for idx, device in enumerate(device_ids(cfg.n_devices)):
        ddir = out / device
        (ddir / "frames").mkdir(parents=True, exist_ok=True)
        arr, windows = _sensor_rows(idx, cfg, rng)
        for s, e in windows:
            gt.anomalies.append({"device": device, "start_ts": START_TS + s, "end_ts": START_TS + e})
        rows = [list(r) for r in arr]

        if cfg.defects:
            # wrong units: battery in millivolts for a block (device 2 and 5),
            # temperature in Fahrenheit for a block (device 3)
            if idx in (1, 4):
                a = int(cfg.seconds * 0.22) + idx * 10
                for r in rows[a : a + 25]:
                    r[10] = r[10] * 1000.0
                gt.wrong_units.append({"device": device, "column": "battery_v", "unit": "mV", "rows": 25})
            if idx == 2:
                a = int(cfg.seconds * 0.55)
                for r in rows[a : a + 30]:
                    r[12] = r[12] * 9 / 5 + 32
                gt.wrong_units.append({"device": device, "column": "temp_c", "unit": "F", "rows": 30})
            # gap: drop a block of rows on every second device
            if idx % 2 == 0:
                g0 = int(cfg.seconds * 0.45) + idx * 7
                glen = 15 + idx
                gt.gaps.append(
                    {"device": device, "start_ts": rows[g0 - 1][0], "end_ts": rows[g0 + glen][0], "missing_rows": glen}
                )
                del rows[g0 : g0 + glen]
            # duplicates: replay a few rows (as after a link retry)
            d0 = int(cfg.seconds * 0.78) - idx * 3
            dup = [list(r) for r in rows[d0 : d0 + 4]]
            rows[d0 + 4 : d0 + 4] = dup
            gt.duplicates.append({"device": device, "rows": 4})

        _write_sensors(ddir / "sensors.csv", device, rows)
        summary["sensor_rows"] += len(rows)

        # flight log
        log_lines = []
        for i in range(0, cfg.seconds, 5):
            ts = START_TS + i
            comp = LOG_COMPONENTS[(i // 5) % len(LOG_COMPONENTS)]
            in_anom = any(s <= i < e for s, e in windows)
            level = "WARN" if in_anom else "INFO"
            msg = "vibration high rms=2.1" if in_anom and comp == "motor" else f"heartbeat seq={i // 5}"
            log_lines.append(f"{_iso(ts)} {level} [{comp}] {msg}")
        if cfg.defects:
            for k, pos in enumerate((len(log_lines) // 10, len(log_lines) // 2)):
                bad = f"\x00\x00garbled-{k} ###" if k == 0 else "truncated line without timestamp"
                log_lines.insert(pos, bad)
                gt.bad_log_lines.append({"device": device, "line_no": pos + 1})
        (ddir / "flight.log").write_text("\n".join(log_lines) + "\n")
        summary["log_lines"] += len(log_lines)

        # frames
        meta = []
        for fi, i in enumerate(range(0, cfg.seconds, cfg.frame_every_s)):
            img = rng.integers(0, 60, (FRAME_SIZE, FRAME_SIZE))
            cx, cy = (fi * 3) % FRAME_SIZE, (fi * 5) % FRAME_SIZE
            img[max(cy - 3, 0) : cy + 3, max(cx - 3, 0) : cx + 3] = 230
            data = _pgm_bytes(img)
            name = f"frame_{fi:04d}.pgm"
            meta.append(
                {
                    "file": name,
                    "ts": float(START_TS + i),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "width": FRAME_SIZE,
                    "height": FRAME_SIZE,
                    "lat": float(arr[i][7]),
                    "lon": float(arr[i][8]),
                }
            )
            if cfg.defects and fi == 7 and idx % 3 == 0:
                data = data[: len(data) // 3]  # truncated upload
                gt.corrupt_files.append({"device": device, "file": f"frames/{name}", "kind": "truncated"})
            elif cfg.defects and fi == 11 and idx % 3 == 1:
                data = data[:-10] + b"\xff" * 10  # bit rot, header fine, checksum wrong
                gt.corrupt_files.append({"device": device, "file": f"frames/{name}", "kind": "checksum"})
            (ddir / "frames" / name).write_bytes(data)
        (ddir / "frames" / "frames.json").write_text(json.dumps(meta, indent=1))
        summary["frames"] += len(meta)
        summary["devices"].append(device)

    (out / "ground_truth.json").write_text(json.dumps(gt.to_dict(), indent=2, default=float))
    summary["ground_truth"] = {k: len(v) for k, v in gt.to_dict().items()}
    return summary
