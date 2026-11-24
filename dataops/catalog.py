"""Metadata catalog in SQLite: dataset versions, files and quality results."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_version (
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    parent TEXT,
    created_at TEXT NOT NULL,
    n_files INTEGER NOT NULL,
    total_bytes INTEGER NOT NULL,
    PRIMARY KEY (name, version)
);
CREATE TABLE IF NOT EXISTS dataset_file (
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    kind TEXT NOT NULL,
    PRIMARY KEY (name, version, path)
);
CREATE TABLE IF NOT EXISTS quality_result (
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    check_name TEXT NOT NULL,
    status TEXT NOT NULL,
    value REAL,
    threshold REAL,
    PRIMARY KEY (name, version, check_name)
);
"""


def kind_of(path: str) -> str:
    if path.endswith(".parquet"):
        return "parquet"
    if path.endswith(".jsonl"):
        return "jsonl"
    if path.endswith(".pgm"):
        return "image"
    if path.endswith(".json"):
        return "json"
    return "other"


class Catalog:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True) if self.path != ":memory:" else None
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)

    def register(self, manifest: dict) -> bool:
        """Register a manifest. Returns False if that version already existed."""
        cur = self.conn.execute(
            "SELECT 1 FROM dataset_version WHERE name=? AND version=?", (manifest["name"], manifest["version"])
        )
        if cur.fetchone():
            return False
        with self.conn:
            self.conn.execute(
                "INSERT INTO dataset_version VALUES (?,?,?,?,?,?)",
                (
                    manifest["name"],
                    manifest["version"],
                    manifest.get("parent"),
                    manifest["created_at"],
                    manifest["n_files"],
                    manifest["total_bytes"],
                ),
            )
            self.conn.executemany(
                "INSERT INTO dataset_file VALUES (?,?,?,?,?,?)",
                [
                    (manifest["name"], manifest["version"], f["path"], f["sha256"], f["bytes"], kind_of(f["path"]))
                    for f in manifest["files"]
                ],
            )
        return True

    def record_quality(self, name: str, version: str, checks: list[dict]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO quality_result VALUES (?,?,?,?,?,?)",
                [(name, version, c["check"], c["status"], c.get("value"), c.get("threshold")) for c in checks],
            )

    def versions(self, name: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT version, parent, created_at, n_files, total_bytes FROM dataset_version WHERE name=? ORDER BY created_at",
            (name,),
        )
        cols = ["version", "parent", "created_at", "n_files", "total_bytes"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def latest(self, name: str) -> dict | None:
        v = self.versions(name)
        return v[-1] if v else None

    def files_by_kind(self, name: str, version: str) -> dict[str, int]:
        cur = self.conn.execute(
            "SELECT kind, COUNT(*) FROM dataset_file WHERE name=? AND version=? GROUP BY kind ORDER BY kind",
            (name, version),
        )
        return dict(cur.fetchall())

    def quality(self, name: str, version: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT check_name, status, value, threshold FROM quality_result WHERE name=? AND version=? ORDER BY check_name",
            (name, version),
        )
        return [dict(zip(["check", "status", "value", "threshold"], r)) for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()
