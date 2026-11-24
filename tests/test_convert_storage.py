import json

import boto3
import pandas as pd
import pyarrow.parquet as pq
import pytest
from moto import mock_aws

from ingest.convert import normalise_sensors
from ingest.schema import RANGES
from ingest.storage import ObjectStore, sha256_file


def test_curated_parquet_has_no_duplicates(converted):
    df = converted["df"]
    assert not df.duplicated(subset=["device_id", "ts"]).any()


def test_curated_is_sorted_per_device(converted):
    for _, g in converted["df"].groupby("device_id"):
        assert g["ts"].is_monotonic_increasing


def test_units_are_normalised(converted, fleet):
    df = converted["df"]
    injected = sum(u["rows"] for u in fleet["gt"]["wrong_units"])
    assert int((df["unit_fixed"] != "").sum()) == injected
    assert df["battery_v"].max() < 30 and df["temp_c"].max() < 85


def test_curated_values_in_range(converted):
    df = converted["df"]
    for col, (lo, hi) in RANGES.items():
        assert df[col].between(lo, hi).all(), col


def test_row_accounting(converted, fleet):
    s = converted["summary"]
    dups = sum(d["rows"] for d in fleet["gt"]["duplicates"])
    assert s["raw_rows"] - s["clean_rows"] == dups


def test_parquet_schema_types(converted):
    path = next((converted["dir"] / "sensors").glob("device_id=*/part-0.parquet"))
    schema = pq.read_schema(path)
    assert str(schema.field("ts").type) == "double"
    assert str(schema.field("device_id").type) in {"string", "large_string"}


def test_logs_converted_to_jsonl(converted, validation):
    total = 0
    for p in (converted["dir"] / "logs").glob("*.jsonl"):
        for line in p.read_text().splitlines():
            rec = json.loads(line)
            assert {"ts", "level", "component", "message", "device_id"} <= rec.keys()
            total += 1
    assert total == converted["summary"]["log_records"]


def test_frame_metadata_table_marks_bad_frames(converted, fleet):
    frames = pd.concat(pd.read_parquet(p) for p in (converted["dir"] / "frames").glob("*/frames.parquet"))
    assert int((frames["status"] != "ok").sum()) == len(fleet["gt"]["corrupt_files"])


def test_normalise_is_idempotent(converted):
    df = converted["df"].drop(columns=["unit_fixed"])
    again = normalise_sensors(df)
    assert (again["unit_fixed"] == "").all()
    assert len(again) == len(df)


@pytest.fixture()
def store(aws_env):
    with mock_aws():
        client = boto3.client("s3", region_name="eu-central-1")
        s = ObjectStore(client, "field-data")
        s.ensure_bucket()
        yield s


def test_push_uploads_all_files(store, converted):
    res = store.push_dir(converted["dir"], "curated")
    n_files = sum(1 for p in converted["dir"].rglob("*") if p.is_file())
    assert res["uploaded"] == n_files
    assert len(store.list_keys("curated/")) == n_files


def test_second_push_is_idempotent(store, converted):
    store.push_dir(converted["dir"], "curated")
    res = store.push_dir(converted["dir"], "curated")
    assert res["uploaded"] == 0 and res["skipped"] > 0


def test_changed_file_is_reuploaded(store, tmp_path):
    (tmp_path / "a.txt").write_text("one")
    store.push_dir(tmp_path, "x")
    (tmp_path / "a.txt").write_text("two")
    assert store.push_dir(tmp_path, "x")["uploaded"] == 1


def test_object_metadata_carries_checksum(store, tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"\x00\x01payload")
    store.push_dir(tmp_path, "p")
    assert store.remote_sha("p/f.bin") == sha256_file(p)


def test_verify_detects_remote_drift(store, tmp_path):
    (tmp_path / "a.txt").write_text("one")
    store.push_dir(tmp_path, "v")
    assert store.verify(tmp_path, "v") == []
    (tmp_path / "a.txt").write_text("changed locally")
    assert store.verify(tmp_path, "v") == ["v/a.txt"]


def test_ensure_bucket_twice_is_safe(store):
    store.ensure_bucket()
    assert "field-data" in {b["Name"] for b in store.client.list_buckets()["Buckets"]}
