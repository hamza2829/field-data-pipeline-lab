"""Schema and plausible ranges for the raw sensor stream."""

from __future__ import annotations

import re

SENSOR_DTYPES = {
    "ts": "float64",
    "device_id": "string",
    "acc_x": "float64",
    "acc_y": "float64",
    "acc_z": "float64",
    "gyro_x": "float64",
    "gyro_y": "float64",
    "gyro_z": "float64",
    "lat": "float64",
    "lon": "float64",
    "alt_m": "float64",
    "battery_v": "float64",
    "battery_pct": "float64",
    "temp_c": "float64",
}

# physically plausible ranges AFTER unit normalisation
RANGES = {
    "acc_x": (-80.0, 80.0),
    "acc_y": (-80.0, 80.0),
    "acc_z": (-80.0, 80.0),
    "gyro_x": (-35.0, 35.0),
    "gyro_y": (-35.0, 35.0),
    "gyro_z": (-35.0, 35.0),
    "lat": (-90.0, 90.0),
    "lon": (-180.0, 180.0),
    "alt_m": (-100.0, 5000.0),
    "battery_v": (0.0, 30.0),
    "battery_pct": (0.0, 100.0),
    "temp_c": (-40.0, 85.0),
}

# unit heuristics: value above the threshold means the device sent another unit
UNIT_RULES = {
    "battery_v": {"above": 100.0, "unit": "mV", "fix": lambda v: v / 1000.0},
    "temp_c": {"above": 85.0, "unit": "F", "fix": lambda v: (v - 32.0) * 5.0 / 9.0},
}

SAMPLE_PERIOD_S = 1.0
GAP_FACTOR = 1.5

LOG_PATTERN = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) (?P<level>INFO|WARN|ERROR) \[(?P<component>[a-z]+)\] (?P<message>.+)$"
)
