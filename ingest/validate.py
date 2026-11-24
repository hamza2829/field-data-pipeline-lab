"""Validation of raw field data: schema, ranges, gaps, duplicates, units, checksums."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ingest.schema import GAP_FACTOR, LOG_PATTERN, RANGES, SAMPLE_PERIOD_S, SENSOR_DTYPES, UNIT_RULES


@dataclass
class Issue:
    device: str
    kind: str  # schema | gap | duplicate | unit | range | checksum | corrupt | log_parse
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"device": self.device, "kind": self.kind, **self.detail}


def read_sensors(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in SENSOR_DTYPES if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df


def check_schema(df: pd.DataFrame, device: str) -> list[Issue]:
    issues = []
    for col, dtype in SENSOR_DTYPES.items():
        if col not in df.columns:
            issues.append(Issue(device, "schema", {"column": col, "problem": "missing"}))
            continue
        if dtype == "float64" and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(Issue(device, "schema", {"column": col, "problem": f"expected numeric, got {df[col].dtype}"}))
    extra = [c for c in df.columns if c not in SENSOR_DTYPES]
    for c in extra:
        issues.append(Issue(device, "schema", {"column": c, "problem": "unexpected"}))
    return issues


def check_duplicates(df: pd.DataFrame, device: str) -> list[Issue]:
    dup = df.duplicated(subset=["device_id", "ts"], keep="first")
    n = int(dup.sum())
    return [Issue(device, "duplicate", {"rows": n})] if n else []


def check_gaps(df: pd.DataFrame, device: str) -> list[Issue]:
    ts = df["ts"].drop_duplicates().sort_values().to_numpy()
    issues = []
    if len(ts) < 2:
        return issues
    diffs = ts[1:] - ts[:-1]
    for i in (diffs > SAMPLE_PERIOD_S * GAP_FACTOR).nonzero()[0]:
        missing = int(round(diffs[i] / SAMPLE_PERIOD_S)) - 1
        issues.append(
            Issue(device, "gap", {"start_ts": float(ts[i]), "end_ts": float(ts[i + 1]), "missing_rows": missing})
        )
    return issues


def check_units(df: pd.DataFrame, device: str) -> list[Issue]:
    issues = []
    for col, rule in UNIT_RULES.items():
        n = int((df[col] > rule["above"]).sum())
        if n:
            issues.append(Issue(device, "unit", {"column": col, "unit": rule["unit"], "rows": n}))
    return issues


def check_ranges(df: pd.DataFrame, device: str) -> list[Issue]:
    """Range check on values that are NOT explained by a unit problem."""
    issues = []
    for col, (lo, hi) in RANGES.items():
        values = df[col]
        if col in UNIT_RULES:
            values = values[values <= UNIT_RULES[col]["above"]]
        bad = values[(values < lo) | (values > hi) | values.isna()]
        if len(bad):
            issues.append(Issue(device, "range", {"column": col, "rows": int(len(bad))}))
    return issues


def parse_pgm(data: bytes) -> tuple[int, int]:
    """Return (width, height) of a binary PGM, raise ValueError if broken."""
    parts = data.split(b"\n", 3)
    if len(parts) < 4 or parts[0] != b"P5":
        raise ValueError("not a P5 PGM")
    w, h = (int(x) for x in parts[1].split())
    if int(parts[2]) != 255:
        raise ValueError("unsupported maxval")
    if len(parts[3]) != w * h:
        raise ValueError(f"payload {len(parts[3])} bytes, expected {w * h}")
    return w, h


def check_frames(frames_dir: Path, device: str) -> tuple[list[Issue], list[dict]]:
    meta = json.loads((frames_dir / "frames.json").read_text())
    issues, rows = [], []
    for m in meta:
        p = frames_dir / m["file"]
        status = "ok"
        if not p.exists():
            status = "missing"
            issues.append(Issue(device, "corrupt", {"file": f"frames/{m['file']}", "problem": "missing"}))
        else:
            data = p.read_bytes()
            try:
                parse_pgm(data)
            except ValueError as exc:
                status = "corrupt"
                issues.append(Issue(device, "corrupt", {"file": f"frames/{m['file']}", "problem": str(exc)}))
            else:
                if hashlib.sha256(data).hexdigest() != m["sha256"]:
                    status = "checksum_mismatch"
                    issues.append(Issue(device, "checksum", {"file": f"frames/{m['file']}"}))
        rows.append({**m, "device_id": device, "status": status})
    return issues, rows


def check_log(path: Path, device: str) -> tuple[list[Issue], list[dict]]:
    issues, records = [], []
    for no, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        m = LOG_PATTERN.match(line)
        if not m:
            issues.append(Issue(device, "log_parse", {"line_no": no}))
            continue
        records.append({"device_id": device, "line_no": no, **m.groupdict()})
    return issues, records


def validate_device(device_dir: str | Path) -> dict:
    d = Path(device_dir)
    device = d.name
    df = read_sensors(d / "sensors.csv")
    issues = []
    issues += check_schema(df, device)
    issues += check_duplicates(df, device)
    issues += check_gaps(df, device)
    issues += check_units(df, device)
    issues += check_ranges(df, device)
    f_issues, frame_rows = check_frames(d / "frames", device)
    l_issues, log_records = check_log(d / "flight.log", device)
    issues += f_issues + l_issues
    return {
        "device": device,
        "rows": int(len(df)),
        "issues": [i.to_dict() for i in issues],
        "frames": frame_rows,
        "log_records": log_records,
    }


def validate_fleet(raw_dir: str | Path) -> dict:
    raw = Path(raw_dir)
    devices = sorted(p for p in raw.iterdir() if p.is_dir())
    results = [validate_device(p) for p in devices]
    issues = [i for r in results for i in r["issues"]]
    counts: dict[str, int] = {}
    for i in issues:
        counts[i["kind"]] = counts.get(i["kind"], 0) + 1
    return {"devices": results, "issues": issues, "issue_counts": counts}
