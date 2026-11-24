import pandas as pd
import pytest

from ingest.validate import (
    check_duplicates,
    check_gaps,
    check_log,
    check_ranges,
    check_schema,
    check_units,
    parse_pgm,
    read_sensors,
)


def _df(rows):
    cols = [
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
    base = [0.0, "d", 0, 0, 9.81, 0, 0, 0, 48.1, 11.5, 60, 15.0, 80, 30]
    data = []
    for r in rows:
        row = list(base)
        for k, v in r.items():
            row[cols.index(k)] = v
        data.append(row)
    return pd.DataFrame(data, columns=cols)


def test_gap_detection_counts_missing_rows():
    df = _df([{"ts": 0.0}, {"ts": 1.0}, {"ts": 6.0}, {"ts": 7.0}])
    gaps = check_gaps(df, "d")
    assert len(gaps) == 1 and gaps[0].detail["missing_rows"] == 4


def test_no_gap_on_regular_stream():
    assert check_gaps(_df([{"ts": float(i)} for i in range(10)]), "d") == []


def test_duplicate_detection():
    df = _df([{"ts": 0.0}, {"ts": 1.0}, {"ts": 1.0}, {"ts": 2.0}])
    assert check_duplicates(df, "d")[0].detail["rows"] == 1


def test_unit_detection_millivolts_and_fahrenheit():
    df = _df([{"ts": 0.0, "battery_v": 15200.0}, {"ts": 1.0, "temp_c": 95.0}, {"ts": 2.0}])
    found = {(i.detail["column"], i.detail["unit"]) for i in check_units(df, "d")}
    assert found == {("battery_v", "mV"), ("temp_c", "F")}


def test_range_check_ignores_unit_errors_but_flags_real_violations():
    df = _df([{"ts": 0.0, "battery_v": 15200.0}, {"ts": 1.0, "lat": 123.0}, {"ts": 2.0, "battery_pct": -3}])
    cols = {i.detail["column"] for i in check_ranges(df, "d")}
    assert cols == {"lat", "battery_pct"}


def test_range_check_flags_nan():
    df = _df([{"ts": 0.0, "alt_m": float("nan")}])
    assert check_ranges(df, "d")[0].detail["column"] == "alt_m"


def test_schema_check_flags_missing_and_extra_columns():
    df = _df([{"ts": 0.0}]).drop(columns=["alt_m"]).assign(rogue=1)
    problems = {(i.detail["column"], i.detail["problem"]) for i in check_schema(df, "d")}
    assert ("alt_m", "missing") in problems and ("rogue", "unexpected") in problems


def test_schema_check_flags_text_in_numeric_column():
    df = _df([{"ts": 0.0}]).astype({"acc_x": "object"})
    df.loc[0, "acc_x"] = "n/a"
    assert any(i.detail["column"] == "acc_x" for i in check_schema(df, "d"))


def test_read_sensors_rejects_missing_columns(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("ts,device_id\n0,d\n")
    with pytest.raises(ValueError):
        read_sensors(p)


def test_parse_pgm_ok_and_truncated():
    good = b"P5\n2 2\n255\n" + bytes([1, 2, 3, 4])
    assert parse_pgm(good) == (2, 2)
    with pytest.raises(ValueError):
        parse_pgm(good[:-1])
    with pytest.raises(ValueError):
        parse_pgm(b"P6\n2 2\n255\n" + bytes(12))


def test_log_parser(tmp_path):
    p = tmp_path / "f.log"
    p.write_text("2026-11-24T10:00:00Z INFO [nav] ok\ngarbage\n2026-11-24T10:00:05Z WARN [motor] vibration\n")
    issues, recs = check_log(p, "d")
    assert len(recs) == 2 and recs[1]["level"] == "WARN" and recs[1]["component"] == "motor"
    assert [i.detail["line_no"] for i in issues] == [2]


def test_fleet_validation_finds_all_issue_kinds(validation):
    assert set(validation["issue_counts"]) == {"duplicate", "gap", "unit", "corrupt", "checksum", "log_parse"}


def test_fleet_has_no_false_range_or_schema_issues(validation):
    assert "range" not in validation["issue_counts"]
    assert "schema" not in validation["issue_counts"]
