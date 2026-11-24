"""MLflow style experiment tracking written as plain local files.

Layout (mirrors the MLflow file store closely enough to be familiar):

    mlruns/<experiment>/<run_id>/meta.json
    mlruns/<experiment>/<run_id>/params.json
    mlruns/<experiment>/<run_id>/metrics.json
    mlruns/<experiment>/<run_id>/tags.json
    mlruns/<experiment>/<run_id>/artifacts/...
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


class Run:
    def __init__(self, root: Path, experiment: str, name: str, run_id: str):
        self.dir = root / experiment / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "artifacts").mkdir(exist_ok=True)
        self.run_id = run_id
        self.params: dict = {}
        self.metrics: dict = {}
        self.tags: dict = {}
        self.meta = {
            "run_id": run_id,
            "experiment": experiment,
            "name": name,
            "start_time": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": "RUNNING",
        }
        self._flush()

    def log_params(self, params: dict) -> None:
        self.params.update({k: v for k, v in params.items()})
        self._flush()

    def log_metrics(self, metrics: dict) -> None:
        self.metrics.update({k: float(v) for k, v in metrics.items()})
        self._flush()

    def set_tags(self, tags: dict) -> None:
        self.tags.update(tags)
        self._flush()

    def log_artifact_text(self, name: str, text: str) -> Path:
        p = self.dir / "artifacts" / name
        p.write_text(text)
        return p

    def end(self, status: str = "FINISHED") -> None:
        self.meta["status"] = status
        self.meta["end_time"] = datetime.now(UTC).isoformat(timespec="seconds")
        self._flush()

    def _flush(self) -> None:
        for fname, obj in (
            ("meta", self.meta),
            ("params", self.params),
            ("metrics", self.metrics),
            ("tags", self.tags),
        ):
            (self.dir / f"{fname}.json").write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))


class Tracker:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def start_run(self, experiment: str, name: str, params: dict | None = None) -> Run:
        seed = json.dumps({"exp": experiment, "name": name, "params": params or {}}, sort_keys=True, default=str)
        run_id = hashlib.sha256(seed.encode()).hexdigest()[:12]
        run = Run(self.root, experiment, name, run_id)
        if params:
            run.log_params(params)
        return run

    def list_runs(self, experiment: str) -> list[dict]:
        out = []
        for d in sorted((self.root / experiment).glob("*/meta.json")):
            run_dir = d.parent
            out.append(
                {
                    **json.loads(d.read_text()),
                    "params": json.loads((run_dir / "params.json").read_text()),
                    "metrics": json.loads((run_dir / "metrics.json").read_text()),
                }
            )
        return out

    def best_run(self, experiment: str, metric: str) -> dict | None:
        runs = [r for r in self.list_runs(experiment) if metric in r["metrics"]]
        return max(runs, key=lambda r: r["metrics"][metric]) if runs else None
