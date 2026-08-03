#!/usr/bin/env python3
"""Build the fail-closed SHA-256 inventory for the source release."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXCLUDED_PARTS = {".git", ".pytest_cache", ".venv", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_FILES = {
    "artifacts/release_manifest.json",
    "artifacts/validation_receipt.json",
}


def included_files(root: Path) -> tuple[Path, ...]:
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        if any(
            part in EXCLUDED_PARTS or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        if relative.as_posix() in EXCLUDED_FILES:
            continue
        files.append(path)
    return tuple(sorted(files))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/release_manifest.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in included_files(root)
    ]
    manifest = {
        "schema_version": "ts-language-foundations-release-manifest-v0.7",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "sha256",
        "root_name": root.name,
        "file_count": len(entries),
        "excluded_files": sorted(EXCLUDED_FILES),
        "files": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifested {len(entries)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
