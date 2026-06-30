#!/usr/bin/env python3
"""Generate SHA-256 checksums for every file in the release tree.

Writes ``SHA256SUMS.txt`` at the release root in the standard
``<hash>  <relative/path>`` format (verifiable with ``sha256sum -c``). The
checksum file itself is excluded. Run last, after package_release.py.
"""
import argparse
import hashlib
from pathlib import Path

SUMS_NAME = "SHA256SUMS.txt"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default="data/release")
    args = ap.parse_args()

    release = Path(args.release)
    files = sorted(
        f for f in release.rglob("*")
        if f.is_file() and f.name != SUMS_NAME and not f.name.startswith(".")
    )
    lines = [f"{sha256(f)}  {f.relative_to(release).as_posix()}" for f in files]
    (release / SUMS_NAME).write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} checksums -> {release / SUMS_NAME}")


if __name__ == "__main__":
    main()
