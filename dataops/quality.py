"""Data quality checks and the quality report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ingest.schema import RANGES, SAMPLE_PERIOD_S


def _status(value: float, warn: float, fail: float, higher_is_better: bool) -> str:
    if higher_is_better:
        return "pass" if value >= warn else ("warn" if value >= fail else "fail")
    return "pass" if value <= warn else ("warn" if value <= fail else "fail")


def run_checks(validation: dict, curated: pd.DataFrame) -> list[dict]:
    checks = []
    issues = validation["issues"]

    # completeness of the sensor stream (curated, after dedup)
    expected = 0
    for _, g in curated.groupby("device_id"):
        expected += int(round((g["ts"].max() - g["ts"].min()) / SAMPLE_PERIOD_S)) + 1
    completeness = 100.0 * len(curated) / expected if expected else 0.0
    checks.append(
        {
            "check": "sensor_completeness_pct",
            "value": round(completeness, 3),
            "threshold": 99.0,
            "status": _status(completeness, 99.0, 95.0, True),
        }
    )

    raw_rows = sum(d["rows"] for d in validation["devices"])
    dup_rows = sum(i["rows"] for i in issues if i["kind"] == "duplicate")
    dup_rate = 100.0 * dup_rows / raw_rows if raw_rows else 0.0
    checks.append(
        {
            "check": "raw_duplicate_rate_pct",
            "value": round(dup_rate, 3),
            "threshold": 0.5,
            "status": _status(dup_rate, 0.5, 2.0, False),
        }
    )
    checks.append(
        {
            "check": "curated_duplicate_rows",
            "value": int(curated.duplicated(subset=["device_id", "ts"]).sum()),
            "threshold": 0,
            "status": "pass" if not curated.duplicated(subset=["device_id", "ts"]).any() else "fail",
        }
    )

    gaps = [i for i in issues if i["kind"] == "gap"]
    checks.append({"check": "gap_count", "value": len(gaps), "threshold": 0, "status": "pass" if not gaps else "warn"})

    unit_rows = sum(i["rows"] for i in issues if i["kind"] == "unit")
    checks.append(
        {"check": "unit_fixed_rows", "value": unit_rows, "threshold": 0, "status": "pass" if not unit_rows else "warn"}
    )

    out_of_range = 0
    for col, (lo, hi) in RANGES.items():
        out_of_range += int(((curated[col] < lo) | (curated[col] > hi) | curated[col].isna()).sum())
    checks.append(
        {
            "check": "curated_out_of_range_values",
            "value": out_of_range,
            "threshold": 0,
            "status": "pass" if out_of_range == 0 else "fail",
        }
    )

    frames = [f for d in validation["devices"] for f in d["frames"]]
    ok = sum(1 for f in frames if f["status"] == "ok")
    frame_ok_pct = 100.0 * ok / len(frames) if frames else 0.0
    checks.append(
        {
            "check": "frame_integrity_pct",
            "value": round(frame_ok_pct, 3),
            "threshold": 99.0,
            "status": _status(frame_ok_pct, 99.0, 90.0, True),
        }
    )

    log_ok = sum(len(d["log_records"]) for d in validation["devices"])
    log_bad = sum(1 for i in issues if i["kind"] == "log_parse")
    log_pct = 100.0 * log_ok / (log_ok + log_bad) if (log_ok + log_bad) else 0.0
    checks.append(
        {
            "check": "log_parse_rate_pct",
            "value": round(log_pct, 3),
            "threshold": 99.0,
            "status": _status(log_pct, 99.0, 95.0, True),
        }
    )

    schema_issues = sum(1 for i in issues if i["kind"] == "schema")
    checks.append(
        {
            "check": "schema_issues",
            "value": schema_issues,
            "threshold": 0,
            "status": "pass" if not schema_issues else "fail",
        }
    )
    return checks


def detection_vs_ground_truth(validation: dict, ground_truth: dict) -> dict:
    """Compare what validation found with what the simulator injected."""
    issues = validation["issues"]

    def found(kind: str, key) -> set:
        return {key(i) for i in issues if i["kind"] == kind}

    res = {}
    gap_found = found("gap", lambda i: (i["device"], i["missing_rows"]))
    gap_true = {(g["device"], g["missing_rows"]) for g in ground_truth["gaps"]}
    res["gaps"] = {
        "injected": len(gap_true),
        "detected": len(gap_true & gap_found),
        "false_positives": len(gap_found - gap_true),
    }

    dup_found = found("duplicate", lambda i: (i["device"], i["rows"]))
    dup_true = {(d["device"], d["rows"]) for d in ground_truth["duplicates"]}
    res["duplicates"] = {
        "injected": len(dup_true),
        "detected": len(dup_true & dup_found),
        "false_positives": len(dup_found - dup_true),
    }

    unit_found = found("unit", lambda i: (i["device"], i["column"], i["rows"]))
    unit_true = {(u["device"], u["column"], u["rows"]) for u in ground_truth["wrong_units"]}
    res["wrong_units"] = {
        "injected": len(unit_true),
        "detected": len(unit_true & unit_found),
        "false_positives": len(unit_found - unit_true),
    }

    corrupt_found = found("corrupt", lambda i: (i["device"], i["file"])) | found(
        "checksum", lambda i: (i["device"], i["file"])
    )
    corrupt_true = {(c["device"], c["file"]) for c in ground_truth["corrupt_files"]}
    res["corrupt_files"] = {
        "injected": len(corrupt_true),
        "detected": len(corrupt_true & corrupt_found),
        "false_positives": len(corrupt_found - corrupt_true),
    }

    log_found = found("log_parse", lambda i: (i["device"], i["line_no"]))
    log_true = {(b["device"], b["line_no"]) for b in ground_truth["bad_log_lines"]}
    res["bad_log_lines"] = {
        "injected": len(log_true),
        "detected": len(log_true & log_found),
        "false_positives": len(log_found - log_true),
    }

    tot_inj = sum(v["injected"] for v in res.values())
    tot_det = sum(v["detected"] for v in res.values())
    res["total"] = {
        "injected": tot_inj,
        "detected": tot_det,
        "false_positives": sum(v["false_positives"] for v in res.values()),
    }
    return res


def render_report(version: str, checks: list[dict], detection: dict, convert_summary: dict) -> str:
    lines = [
        "# Data quality report",
        "",
        f"Dataset version: `{version}`",
        "",
        "## Checks",
        "",
        "| Check | Value | Threshold | Status |",
        "|---|---:|---:|---|",
    ]
    for c in checks:
        lines.append(f"| {c['check']} | {c['value']} | {c['threshold']} | {c['status']} |")
    lines += [
        "",
        "## Injected defects vs detected",
        "",
        "| Defect type | Injected | Detected | False positives |",
        "|---|---:|---:|---:|",
    ]
    for k, v in detection.items():
        lines.append(f"| {k} | {v['injected']} | {v['detected']} | {v['false_positives']} |")
    lines += [
        "",
        "## Conversion",
        "",
        f"- raw sensor rows: {convert_summary['raw_rows']}, curated rows: {convert_summary['clean_rows']}",
        f"- rows with unit correction: {convert_summary['unit_fixed_rows']}",
        f"- structured log records: {convert_summary['log_records']}",
        f"- frames intact: {convert_summary['frames_ok']} of {convert_summary['frames']}",
        f"- sensor CSV {convert_summary['csv_bytes']} bytes -> Parquet {convert_summary['parquet_bytes']} bytes "
        f"(ratio {convert_summary['compression_ratio']})",
        "",
    ]
    return "\n".join(lines)


def write_report(path: str | Path, text: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    p.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str))
