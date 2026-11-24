import json

import pipeline


def _run_all(work):
    assert pipeline.main(["all", "--work", str(work), "--devices", "4", "--seconds", "600"]) == 0
    return {p.stem: json.loads(p.read_text()) for p in (work / "state").glob("*.json")}


def test_pipeline_end_to_end(tmp_path, aws_env):
    state = _run_all(tmp_path / "w")
    assert state["ingest"]["mode"] == "moto"
    assert state["ingest"]["second_push_uploaded"] == 0
    assert state["ingest"]["checksum_mismatches"] == 0
    assert state["version"]["checks_failed"] == []
    assert state["version"]["detection_total"]["detected"] == state["version"]["detection_total"]["injected"]
    assert (tmp_path / "w" / "report" / "quality_report.md").exists()
    assert (tmp_path / "w" / "report" / "model_comparison.md").exists()
    assert (tmp_path / "w" / "catalog.sqlite").exists()


def test_pipeline_is_reproducible(tmp_path, aws_env):
    a = _run_all(tmp_path / "a")
    b = _run_all(tmp_path / "b")
    assert a["version"]["version"] == b["version"]["version"]
    assert a["evaluate"]["models"] == b["evaluate"]["models"]


def test_rerun_does_not_create_new_version(tmp_path, aws_env):
    work = tmp_path / "w"
    _run_all(work)
    first = json.loads((work / "state" / "version.json").read_text())
    pipeline.main(["version", "--work", str(work)])
    second = json.loads((work / "state" / "version.json").read_text())
    assert second["version"] == first["version"] and second["new_version"] is False
