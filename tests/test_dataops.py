import pandas as pd

from dataops.catalog import Catalog, kind_of
from dataops.quality import detection_vs_ground_truth, render_report, run_checks
from dataops.versioning import build_manifest, diff_manifests, verify_manifest, version_id, write_manifest


def test_version_id_is_content_based(tmp_path):
    (tmp_path / "a").write_text("x")
    m1 = build_manifest(tmp_path, "ds")
    m2 = build_manifest(tmp_path, "ds")
    assert m1["version"] == m2["version"]
    (tmp_path / "a").write_text("y")
    assert build_manifest(tmp_path, "ds")["version"] != m1["version"]


def test_version_id_independent_of_entry_order():
    e = [{"path": "b", "sha256": "2"}, {"path": "a", "sha256": "1"}]
    assert version_id(e) == version_id(list(reversed(e)))


def test_version_id_changes_on_rename():
    assert version_id([{"path": "a", "sha256": "1"}]) != version_id([{"path": "b", "sha256": "1"}])


def test_manifest_diff(tmp_path):
    (tmp_path / "keep").write_text("k")
    (tmp_path / "change").write_text("c1")
    (tmp_path / "gone").write_text("g")
    old = build_manifest(tmp_path, "ds")
    (tmp_path / "change").write_text("c2")
    (tmp_path / "gone").unlink()
    (tmp_path / "new").write_text("n")
    d = diff_manifests(old, build_manifest(tmp_path, "ds"))
    assert d == {"added": ["new"], "removed": ["gone"], "changed": ["change"]}


def test_verify_manifest_detects_tampering(tmp_path):
    (tmp_path / "a").write_text("x")
    m = build_manifest(tmp_path, "ds")
    assert verify_manifest(m, tmp_path) == []
    (tmp_path / "a").write_text("tampered")
    assert verify_manifest(m, tmp_path) == ["a"]


def test_write_manifest_path(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f").write_text("1")
    m = build_manifest(tmp_path / "d", "ds")
    p = write_manifest(m, tmp_path / "versions")
    assert p == tmp_path / "versions" / "ds" / m["version"] / "manifest.json"


def test_catalog_register_is_idempotent(converted):
    cat = Catalog(":memory:")
    m = build_manifest(converted["dir"], "curated")
    assert cat.register(m) is True
    assert cat.register(m) is False
    assert len(cat.versions("curated")) == 1


def test_catalog_lineage_and_kinds(converted, tmp_path):
    cat = Catalog(tmp_path / "c.sqlite")
    m1 = build_manifest(converted["dir"], "curated", created_at="2026-01-01T00:00:00+00:00")
    cat.register(m1)
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "extra.json").write_text("{}")
    m2 = build_manifest(tmp_path / "x", "curated", parent=m1["version"], created_at="2026-01-02T00:00:00+00:00")
    cat.register(m2)
    assert cat.latest("curated")["parent"] == m1["version"]
    kinds = cat.files_by_kind("curated", m1["version"])
    assert kinds["parquet"] == 12 and kinds["jsonl"] == 6


def test_catalog_quality_roundtrip():
    cat = Catalog(":memory:")
    cat.record_quality("ds", "v1", [{"check": "c", "status": "pass", "value": 1.0, "threshold": 0.0}])
    assert cat.quality("ds", "v1")[0]["status"] == "pass"


def test_kind_of():
    assert [kind_of(p) for p in ["a.parquet", "b.jsonl", "c.pgm", "d.json", "e.txt"]] == [
        "parquet",
        "jsonl",
        "image",
        "json",
        "other",
    ]


def test_all_injected_defects_are_detected(validation, fleet):
    det = detection_vs_ground_truth(validation, fleet["gt"])
    assert det["total"]["detected"] == det["total"]["injected"]
    assert det["total"]["false_positives"] == 0


def test_quality_checks_on_curated_data(validation, converted):
    checks = {c["check"]: c for c in run_checks(validation, converted["df"])}
    assert checks["curated_duplicate_rows"]["status"] == "pass"
    assert checks["curated_out_of_range_values"]["status"] == "pass"
    assert checks["gap_count"]["status"] == "warn"
    assert checks["frame_integrity_pct"]["value"] < 100


def test_quality_check_fails_on_out_of_range(validation, converted):
    bad = converted["df"].copy()
    bad.loc[0, "lat"] = 999.0
    checks = {c["check"]: c for c in run_checks(validation, bad)}
    assert checks["curated_out_of_range_values"]["status"] == "fail"


def test_report_renders_tables(validation, converted, fleet):
    checks = run_checks(validation, converted["df"])
    det = detection_vs_ground_truth(validation, fleet["gt"])
    text = render_report("abc", checks, det, converted["summary"])
    assert "| sensor_completeness_pct |" in text and "| total |" in text and "`abc`" in text
    assert isinstance(converted["df"], pd.DataFrame)
