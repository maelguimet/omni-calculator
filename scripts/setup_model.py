#!/usr/bin/env python3
"""Fetch a pinned, verified Needle engine for bundling. Build-time only; stdlib only."""
import argparse
import hashlib
import io
import os
import pathlib
import platform
import tempfile
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
REVISION = "32e9e3a93b205f786929697446ae669cf0a84579"
WHEEL = "python/cactus_needle-2.0.4-py3-none-manylinux2014_x86_64.whl"
WHEEL_SHA = "13a84e6c73095fd175b11d46a30a984b62123d94421b769c107074aff7f65c2b"
LIB_SHA = "9fa5386d3e3a8ee17914fb23643bc5f5c906b683fa33561e79c5445dd78bc389"
DEST = ROOT / ".cache" / "needle" / "libneedle.so"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Only verify the cached engine")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        parser.error("This release bundles the Linux x86-64 engine; other architectures are not packaged yet.")
    if DEST.is_file() and digest(DEST.read_bytes()) == LIB_SHA:
        print(f"Verified {DEST}")
        return
    if args.offline:
        parser.error("The verified engine is not cached. Run this script once with network access.")
    url = f"https://huggingface.co/Cactus-Compute/needle2/resolve/{REVISION}/{WHEEL}?download=true"
    print("Downloading pinned Needle 2 engine (about 13 MB)...", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(20 * 1024 * 1024 + 1)
    if digest(data) != WHEEL_SHA:
        raise RuntimeError("Engine package checksum mismatch; refusing to install it")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        library = archive.read("needle/libneedle.so")
    if digest(library) != LIB_SHA:
        raise RuntimeError("Native library checksum mismatch")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=DEST.parent, prefix=".download-", delete=False) as output:
            temporary = pathlib.Path(output.name)
            output.write(library)
        os.replace(temporary, DEST)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Verified {DEST} ({len(library):,} bytes)")


if __name__ == "__main__":
    main()
