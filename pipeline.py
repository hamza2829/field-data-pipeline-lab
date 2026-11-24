"""Command line entry point for the whole DataOps/MLOps pipeline.

    python pipeline.py all --work build
    python pipeline.py generate|ingest|validate|convert|version|evaluate --work build

Object storage: by default an in-process moto S3 mock is used, so the pipeline
needs no account. Set S3_ENDPOINT_URL (for example the MinIO service from
docker/docker-compose.yml) to push to a real S3 compatible endpoint instead.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

from simulator.fleet import FleetConfig, simulate_fleet

BUCKET = "field-data"


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def _load(path: Path):
    return json.loads(path.read_text())


@contextlib.contextmanager
def object_store():
    from ingest.storage import ObjectStore, make_client

    if os.environ.get("S3_ENDPOINT_URL"):
        store = ObjectStore(make_client(), BUCKET)
        store.ensure_bucket()
        yield store, "endpoint"
        return
    from moto import mock_aws

    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        store = ObjectStore(make_client(), BUCKET)
        store.ensure_bucket()
        yield store, "moto"


def stage_generate(work: Path, args) -> dict:
    cfg = FleetConfig(n_devices=args.devices, seconds=args.seconds, seed=args.seed)
    summary = simulate_fleet(work / "raw", cfg)
    _dump(work / "state" / "generate.json", summary)
    return summary


def stage_ingest(work: Path, args) -> dict:
    """Land the raw files in object storage (raw zone) and verify checksums."""
    with object_store() as (store, mode):
        res = store.push_dir(work / "raw", "raw")
        again = store.push_dir(work / "raw", "raw")  # idempotency check
        bad = store.verify(work / "raw", "raw")
        res.update(
            {
                "mode": mode,
                "second_push_uploaded": again["uploaded"],
                "checksum_mismatches": len(bad),
                "objects": len(store.list_keys("raw/")),
            }
        )
    _dump(work / "state" / "ingest.json", res)
    return res


def stage_validate(work: Path, args) -> dict:
    from ingest.validate import validate_fleet

    val = validate_fleet(work / "raw")
    _dump(work / "state" / "validation.json", val)
    return {"issue_counts": val["issue_counts"], "issues": len(val["issues"])}


def stage_convert(work: Path, args) -> dict:
    from ingest.convert import convert_fleet

    val = _load(work / "state" / "validation.json")
    res = convert_fleet(work / "raw", work / "curated", val)
    with object_store() as (store, mode):
        push = store.push_dir(work / "curated", "curated")
    res["push"] = {**push, "mode": mode}
    _dump(work / "state" / "convert.json", res)
    return {k: v for k, v in res.items() if k != "devices"}


def stage_version(work: Path, args) -> dict:
    from dataops.catalog import Catalog
    from dataops.quality import detection_vs_ground_truth, render_report, run_checks, write_report
    from dataops.versioning import build_manifest, write_manifest
    from ingest.convert import read_curated_sensors

    cat = Catalog(work / "catalog.sqlite")
    parent = cat.latest("curated")
    manifest = build_manifest(work / "curated", "curated", parent["version"] if parent else None)
    if parent and parent["version"] == manifest["version"]:
        manifest["parent"] = parent["parent"]
    write_manifest(manifest, work / "versions")
    new = cat.register(manifest)

    val = _load(work / "state" / "validation.json")
    conv = _load(work / "state" / "convert.json")
    gt = _load(work / "raw" / "ground_truth.json")
    curated = read_curated_sensors(work / "curated")
    checks = run_checks(val, curated)
    detection = detection_vs_ground_truth(val, gt)
    cat.record_quality("curated", manifest["version"], checks)
    text = render_report(manifest["version"], checks, detection, conv)
    write_report(work / "report" / "quality_report.md", text, {"checks": checks, "detection": detection})
    res = {
        "version": manifest["version"],
        "new_version": new,
        "n_files": manifest["n_files"],
        "total_bytes": manifest["total_bytes"],
        "files_by_kind": cat.files_by_kind("curated", manifest["version"]),
        "checks_failed": [c["check"] for c in checks if c["status"] == "fail"],
        "detection_total": detection["total"],
    }
    cat.close()
    _dump(work / "state" / "version.json", res)
    return res


def stage_evaluate(work: Path, args) -> dict:
    from ingest.convert import read_curated_sensors
    from mlops.evaluate import render_comparison, run_evaluation

    version = _load(work / "state" / "version.json")["version"]
    gt = _load(work / "raw" / "ground_truth.json")
    curated = read_curated_sensors(work / "curated")
    summary = run_evaluation(curated, gt["anomalies"], work / "mlruns", version, work / "datasets" / version)
    (work / "report").mkdir(parents=True, exist_ok=True)
    (work / "report" / "model_comparison.md").write_text(render_comparison(summary))
    _dump(work / "state" / "evaluate.json", summary)
    return summary


STAGES = {
    "generate": stage_generate,
    "ingest": stage_ingest,
    "validate": stage_validate,
    "convert": stage_convert,
    "version": stage_version,
    "evaluate": stage_evaluate,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*STAGES, "all"])
    ap.add_argument("--work", default="build")
    ap.add_argument("--devices", type=int, default=6)
    ap.add_argument("--seconds", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    work = Path(args.work)
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    timings = {}
    for name in stages:
        t0 = time.perf_counter()
        res = STAGES[name](work, args)
        timings[name] = round(time.perf_counter() - t0, 3)
        print(f"== {name} ({timings[name]} s)")
        print(json.dumps(res, indent=1, default=str)[:2000])
    if args.stage == "all":
        _dump(work / "state" / "timings.json", timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
