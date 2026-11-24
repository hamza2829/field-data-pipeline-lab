import hashlib
import json

from simulator.fleet import SENSOR_COLUMNS, FleetConfig, device_ids, simulate_fleet


def test_device_ids_are_stable():
    assert device_ids(3) == ["drone-01", "drone-02", "drone-03"]


def test_every_device_has_all_raw_artifacts(fleet):
    for dev in device_ids(6):
        d = fleet["raw"] / dev
        assert (d / "sensors.csv").exists()
        assert (d / "flight.log").exists()
        assert (d / "frames" / "frames.json").exists()


def test_sensor_header_matches_schema(fleet):
    header = (fleet["raw"] / "drone-01" / "sensors.csv").read_text().splitlines()[0]
    assert header.split(",") == SENSOR_COLUMNS


def test_same_seed_gives_identical_bytes(tmp_path):
    cfg = FleetConfig(n_devices=2, seconds=120, seed=5, anomalies_per_device=1)
    simulate_fleet(tmp_path / "a", cfg)
    simulate_fleet(tmp_path / "b", cfg)
    for rel in ["drone-01/sensors.csv", "drone-02/flight.log", "ground_truth.json"]:
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()


def test_different_seed_changes_data(tmp_path):
    simulate_fleet(tmp_path / "a", FleetConfig(n_devices=1, seconds=120, seed=1, anomalies_per_device=1))
    simulate_fleet(tmp_path / "b", FleetConfig(n_devices=1, seconds=120, seed=2, anomalies_per_device=1))
    assert (tmp_path / "a/drone-01/sensors.csv").read_bytes() != (tmp_path / "b/drone-01/sensors.csv").read_bytes()


def test_ground_truth_lists_every_defect_type(fleet):
    gt = fleet["gt"]
    for key in ["gaps", "duplicates", "wrong_units", "corrupt_files", "bad_log_lines", "anomalies"]:
        assert gt[key], key
    assert len(gt["anomalies"]) == 6 * 3


def test_clean_mode_injects_no_defects(tmp_path):
    simulate_fleet(tmp_path, FleetConfig(n_devices=3, seconds=200, defects=False, anomalies_per_device=1))
    gt = json.loads((tmp_path / "ground_truth.json").read_text())
    assert not gt["gaps"] and not gt["duplicates"] and not gt["corrupt_files"] and not gt["bad_log_lines"]


def test_frame_metadata_checksums_match_intact_frames(fleet):
    d = fleet["raw"] / "drone-03" / "frames"  # idx 2: no corrupted frame injected
    meta = json.loads((d / "frames.json").read_text())
    for m in meta:
        assert hashlib.sha256((d / m["file"]).read_bytes()).hexdigest() == m["sha256"]


def test_summary_counts(fleet):
    s = fleet["summary"]
    assert s["devices"] == device_ids(6)
    assert s["frames"] == 6 * 30
    assert s["sensor_rows"] > 5000
