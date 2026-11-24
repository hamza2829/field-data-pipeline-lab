"""Dataset versioning by content hash.

A dataset version id is the SHA-256 over the sorted list of (relative path,
file hash) pairs. The same bytes always give the same id, and any change to
any file gives a new one.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ingest.storage import sha256_file


def file_entries(data_dir: str | Path) -> list[dict]:
    root = Path(data_dir)
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out.append({"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    return out


def version_id(entries: list[dict]) -> str:
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda x: x["path"]):
        h.update(f"{e['path']}\0{e['sha256']}\n".encode())
    return h.hexdigest()[:16]


def build_manifest(data_dir: str | Path, name: str, parent: str | None = None, created_at: str | None = None) -> dict:
    entries = file_entries(data_dir)
    return {
        "name": name,
        "version": version_id(entries),
        "parent": parent,
        "created_at": created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "n_files": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "files": entries,
    }


def write_manifest(manifest: dict, versions_dir: str | Path) -> Path:
    d = Path(versions_dir) / manifest["name"] / manifest["version"]
    d.mkdir(parents=True, exist_ok=True)
    p = d / "manifest.json"
    p.write_text(json.dumps(manifest, indent=1))
    return p


def diff_manifests(old: dict, new: dict) -> dict:
    a = {e["path"]: e["sha256"] for e in old["files"]}
    b = {e["path"]: e["sha256"] for e in new["files"]}
    return {
        "added": sorted(set(b) - set(a)),
        "removed": sorted(set(a) - set(b)),
        "changed": sorted(k for k in set(a) & set(b) if a[k] != b[k]),
    }


def verify_manifest(manifest: dict, data_dir: str | Path) -> list[str]:
    """Return paths that are missing or whose content changed."""
    root = Path(data_dir)
    bad = []
    for e in manifest["files"]:
        p = root / e["path"]
        if not p.exists() or sha256_file(p) != e["sha256"]:
            bad.append(e["path"])
    return bad
