# Run book

## Run everything locally

```bash
python -m pip install -r requirements-dev.txt
python pipeline.py all --work build        # about 5 s on a laptop class CPU
pytest -q
```

Outputs: `build/report/quality_report.md`, `build/report/model_comparison.md`,
`build/catalog.sqlite`, `build/versions/`, `build/mlruns/`.

## Run single stages (as CI does)

```bash
python pipeline.py generate --work build   # --devices 6 --seconds 900 --seed 42
python pipeline.py ingest   --work build   # raw zone -> S3 (moto unless S3_ENDPOINT_URL is set)
python pipeline.py validate --work build
python pipeline.py convert  --work build   # curated zone -> S3
python pipeline.py version  --work build   # manifest, catalog, quality report
python pipeline.py evaluate --work build   # features, split, two models, tracking
```

Every stage writes its result to `build/state/<stage>.json`.

## Against a real S3 compatible endpoint

```bash
export S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
python pipeline.py ingest --work build
```

Or the whole edge setup: `docker compose -f docker/docker-compose.yml up --build`.

## Typical alerts and what to do

| Symptom | Where you see it | Action |
|---|---|---|
| `gap_count` warn | quality report | check the device's link log around the gap timestamps; data is not invented, the gap stays |
| `raw_duplicate_rate_pct` above 0.5 % | quality report | link retries; curated data is already deduplicated, look at the uplink |
| `unit_fixed_rows` warn | quality report | a device sends mV or °F; rows are converted, open a firmware ticket with the device id from `issues` |
| `frame_integrity_pct` below 99 | quality report | `frames.parquet` lists frames with status `corrupt` or `checksum_mismatch`; request re-upload |
| `curated_out_of_range_values` fail | quality report, CI gate | a real sensor fault or a new unit problem; stop promotion of that version |
| `checksum_mismatches` > 0 after ingest | `state/ingest.json` | object store copy differs from source; push again, then verify |
| new dataset version on a rerun without new data | `state/version.json` `new_version: true` | something non deterministic entered the curated zone (library upgrade changes Parquet bytes); compare manifests with `diff_manifests` |

## Roll back to a previous dataset version

1. `sqlite3 build/catalog.sqlite "select version, parent, created_at from dataset_version"`
2. Read `build/versions/curated/<version>/manifest.json`.
3. Restore the listed files from the object store (keys are `curated/<path>`),
   then `dataops.versioning.verify_manifest` must return an empty list.

## Adding a new check

Add it to `dataops/quality.py::run_checks` with a threshold and a status,
add a test in `tests/test_dataops.py`, and if it should block a release add
it to the "Gate on results" step in `.github/workflows/ci.yml`.
