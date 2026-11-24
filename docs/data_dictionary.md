# Data dictionary

## Raw zone (`raw/<device>/...`)

### sensors.csv (1 Hz)

| Column | Type | Unit | Valid range | Notes |
|---|---|---|---|---|
| ts | float | s since Unix epoch, UTC | - | one row per second; gaps and replays happen |
| device_id | string | - | `drone-NN` | |
| acc_x, acc_y, acc_z | float | m/s² | -80 .. 80 | IMU accelerometer, z includes gravity |
| gyro_x, gyro_y, gyro_z | float | rad/s | -35 .. 35 | IMU gyroscope |
| lat, lon | float | degrees WGS84 | ±90, ±180 | GPS |
| alt_m | float | m | -100 .. 5000 | altitude above take off |
| battery_v | float | V | 0 .. 30 | some firmware sends mV; values > 100 are treated as mV |
| battery_pct | float | % | 0 .. 100 | |
| temp_c | float | °C | -40 .. 85 | some firmware sends °F; values > 85 are treated as °F |

### flight.log

One line per event: `YYYY-MM-DDTHH:MM:SSZ LEVEL [component] message`,
LEVEL in INFO, WARN, ERROR; component in nav, motor, power, camera, link.
Lines that do not match are counted as parse failures and kept out of the
curated zone.

### frames/frame_NNNN.pgm and frames/frames.json

Binary PGM (P5), 32 x 32 greyscale. `frames.json` lists per frame: `file`,
`ts`, `sha256`, `width`, `height`, `lat`, `lon`.

### ground_truth.json (simulator only)

Lists every injected defect and every injected motor fault window. Used by
tests and the quality report to measure detection; never used for training.

## Curated zone (`curated/...`)

| Path | Format | Content |
|---|---|---|
| `sensors/device_id=<id>/part-0.parquet` | Parquet, zstd | sensor rows, deduplicated, sorted by ts, units normalised, plus `unit_fixed` (e.g. `battery_v:mV;`) |
| `logs/<id>.jsonl` | JSON lines | `device_id`, `line_no`, `ts`, `level`, `component`, `message` |
| `frames/device_id=<id>/frames.parquet` | Parquet | frame metadata plus `status` in ok, corrupt, checksum_mismatch, missing |

## DataOps artefacts

| Path | Content |
|---|---|
| `versions/curated/<version>/manifest.json` | file list with SHA-256 and size; version id = first 16 hex chars of SHA-256 over sorted `(path, sha256)` |
| `catalog.sqlite` | tables `dataset_version`, `dataset_file`, `quality_result` |
| `report/quality_report.md` + `.json` | checks, injected vs detected defects, conversion numbers |

## MLOps artefacts

| Path | Content |
|---|---|
| `datasets/<version>/train.parquet`, `val.parquet` | 10 s window features with label, split by device |
| `mlruns/drone-anomaly/<run_id>/` | `meta.json`, `params.json` (includes `dataset_version`), `metrics.json`, `tags.json`, `artifacts/` |
| `report/model_comparison.md` | precision, recall, F1, ROC AUC, average precision per model |

### Window features

| Feature | Definition |
|---|---|
| acc_rms | RMS of acceleration magnitude after removing 9.81 m/s² on z |
| acc_std | standard deviation of that magnitude |
| gyro_rms | RMS of gyro magnitude |
| battery_v_mean | mean battery voltage |
| battery_v_slope | least squares slope of battery voltage over the window (V/s) |
| temp_mean | mean temperature |
| temp_delta | temp_mean minus the device's median temp_mean |
