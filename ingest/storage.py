"""Push curated data to S3 compatible object storage.

The store only needs a boto3 S3 client, so it works against AWS S3, MinIO
(docker compose edge setup) or moto (tests and the CI pipeline run). Objects
are keyed by path and carry their SHA-256 as metadata, which makes uploads
idempotent: an unchanged file is skipped on the next push.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def make_client(endpoint_url: str | None = None):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or os.environ.get("S3_ENDPOINT_URL"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-central-1"),
    )


class ObjectStore:
    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket

    def ensure_bucket(self) -> None:
        existing = {b["Name"] for b in self.client.list_buckets().get("Buckets", [])}
        if self.bucket not in existing:
            region = self.client.meta.region_name
            if region and region != "us-east-1":
                self.client.create_bucket(Bucket=self.bucket, CreateBucketConfiguration={"LocationConstraint": region})
            else:
                self.client.create_bucket(Bucket=self.bucket)

    def remote_sha(self, key: str) -> str | None:
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.ClientError:
            return None
        return head.get("Metadata", {}).get("sha256")

    def put_file(self, path: Path, key: str) -> bool:
        """Upload if the remote copy differs. Returns True when uploaded."""
        digest = sha256_file(path)
        if self.remote_sha(key) == digest:
            return False
        with open(path, "rb") as fh:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=fh, Metadata={"sha256": digest})
        return True

    def push_dir(self, local_dir: str | Path, prefix: str) -> dict:
        root = Path(local_dir)
        uploaded = skipped = nbytes = 0
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            key = f"{prefix.rstrip('/')}/{p.relative_to(root).as_posix()}"
            if self.put_file(p, key):
                uploaded += 1
                nbytes += p.stat().st_size
            else:
                skipped += 1
        return {"uploaded": uploaded, "skipped": skipped, "bytes": nbytes}

    def list_keys(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys += [o["Key"] for o in page.get("Contents", [])]
        return keys

    def verify(self, local_dir: str | Path, prefix: str) -> list[str]:
        """Return keys whose remote checksum does not match the local file."""
        root = Path(local_dir)
        bad = []
        for p in sorted(root.rglob("*")):
            if p.is_file():
                key = f"{prefix.rstrip('/')}/{p.relative_to(root).as_posix()}"
                if self.remote_sha(key) != sha256_file(p):
                    bad.append(key)
        return bad
