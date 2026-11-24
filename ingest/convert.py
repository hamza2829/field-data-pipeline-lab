"""Conversion of raw field data into curated, analysis ready formats.

* sensors.csv -> Parquet (deduplicated, sorted, units normalised, flags kept)
* flight.log  -> JSON lines (one structured record per parseable line)
* frames.json -> Parquet metadata table with a status per frame
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ingest.schema import SENSOR_DTYPES, UNIT_RULES
from ingest.validate import read_sensors, validate_device


def normalise_sensors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates(subset=["device_id", "ts"], keep="first")
    out = out.sort_values(["device_id", "ts"]).reset_index(drop=True)
    out["unit_fixed"] = ""
    for col, rule in UNIT_RULES.items():
        mask = out[col] > rule["above"]
        out.loc[mask, col] = out.loc[mask, col].map(rule["fix"])
        out.loc[mask, "unit_fixed"] = out.loc[mask, "unit_fixed"] + f"{col}:{rule['unit']};"
    for col, dtype in SENSOR_DTYPES.items():
        out[col] = out[col].astype("string" if dtype == "string" else dtype)
    out["unit_fixed"] = out["unit_fixed"].astype("string")
    return out


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def convert_device(device_dir: str | Path, curated_dir: str | Path, validation: dict | None = None) -> dict:
    d = Path(device_dir)
    cur = Path(curated_dir)
    device = d.name
    validation = validation or validate_device(d)

    raw = read_sensors(d / "sensors.csv")
    clean = normalise_sensors(raw)
    sensor_path = cur / "sensors" / f"device_id={device}" / "part-0.parquet"
    write_parquet(clean, sensor_path)

    log_path = cur / "logs" / f"{device}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        for rec in validation["log_records"]:
            fh.write(json.dumps(rec) + "\n")

    frames = pd.DataFrame(validation["frames"])
    frames_path = cur / "frames" / f"device_id={device}" / "frames.parquet"
    write_parquet(frames, frames_path)

    return {
        "device": device,
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "unit_fixed_rows": int((clean["unit_fixed"] != "").sum()),
        "log_records": len(validation["log_records"]),
        "frames": int(len(frames)),
        "frames_ok": int((frames["status"] == "ok").sum()) if len(frames) else 0,
        "csv_bytes": (d / "sensors.csv").stat().st_size,
        "parquet_bytes": sensor_path.stat().st_size,
    }


def convert_fleet(raw_dir: str | Path, curated_dir: str | Path, validation: dict | None = None) -> dict:
    raw = Path(raw_dir)
    by_dev = {r["device"]: r for r in (validation or {}).get("devices", [])}
    results = [convert_device(p, curated_dir, by_dev.get(p.name)) for p in sorted(raw.iterdir()) if p.is_dir()]
    total_csv = sum(r["csv_bytes"] for r in results)
    total_pq = sum(r["parquet_bytes"] for r in results)
    return {
        "devices": results,
        "raw_rows": sum(r["raw_rows"] for r in results),
        "clean_rows": sum(r["clean_rows"] for r in results),
        "unit_fixed_rows": sum(r["unit_fixed_rows"] for r in results),
        "log_records": sum(r["log_records"] for r in results),
        "frames_ok": sum(r["frames_ok"] for r in results),
        "frames": sum(r["frames"] for r in results),
        "csv_bytes": total_csv,
        "parquet_bytes": total_pq,
        "compression_ratio": round(total_csv / total_pq, 2) if total_pq else None,
    }


def read_curated_sensors(curated_dir: str | Path) -> pd.DataFrame:
    parts = sorted((Path(curated_dir) / "sensors").glob("device_id=*/part-0.parquet"))
    return pd.concat([pq.read_table(p).to_pandas() for p in parts], ignore_index=True)
