import json
import os

import pytest

from ingest.convert import convert_fleet, read_curated_sensors
from ingest.validate import validate_fleet
from simulator.fleet import FleetConfig, simulate_fleet


@pytest.fixture(scope="session")
def fleet(tmp_path_factory):
    root = tmp_path_factory.mktemp("fleet")
    raw = root / "raw"
    summary = simulate_fleet(raw, FleetConfig(n_devices=6, seconds=900, seed=42))
    gt = json.loads((raw / "ground_truth.json").read_text())
    return {"root": root, "raw": raw, "summary": summary, "gt": gt}


@pytest.fixture(scope="session")
def validation(fleet):
    return validate_fleet(fleet["raw"])


@pytest.fixture(scope="session")
def converted(fleet, validation):
    cur = fleet["root"] / "curated"
    summary = convert_fleet(fleet["raw"], cur, validation)
    return {"dir": cur, "summary": summary, "df": read_curated_sensors(cur)}


@pytest.fixture()
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    yield os.environ
