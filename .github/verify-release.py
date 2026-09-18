#!/usr/bin/env python3
"""Independently check the release archive, then exercise its actual executable."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    dist = args.dist.resolve(strict=True)
    manifest = json.loads((dist / "manifest.json").read_text())
    name = f"omni-calc-{manifest['version']}-linux-x86_64"
    expected_assets = {"calc", "manifest.json", name + ".tar.gz"}
    checked_assets = set()
    for line in (dist / "SHA256SUMS").read_text().splitlines():
        expected_sha, filename = line.split("  ", 1)
        require(filename in expected_assets, f"Unexpected checksum entry: {filename}")
        require(filename not in checked_assets, f"Duplicate checksum entry: {filename}")
        require(sha(dist / filename) == expected_sha, f"Checksum mismatch: {filename}")
        checked_assets.add(filename)
    require(checked_assets == expected_assets, "Missing release checksums")
    require(manifest["source_revision"] == args.expected_revision,
            "Packaged source revision differs from the tested GitHub commit")

    # The archive, its manifest, and the standalone assets must agree exactly.
    with tempfile.TemporaryDirectory(prefix="omni-release-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        extraction = Path(temporary)
        with tarfile.open(dist / (name + ".tar.gz"), "r:gz") as archive:
            members = archive.getmembers()
            member_names = {member.name for member in members}
            require(len(member_names) == len(members), "Duplicate archive members")
            for member in members:
                path = PurePosixPath(member.name)
                require(member.isfile() and not path.is_absolute()
                        and ".." not in path.parts and path.parts[0] == name,
                        f"Unexpected archive member: {member.name}")
            archive.extractall(extraction, filter="data")
        package = extraction / name
        require((package / "manifest.json").read_bytes() == (dist / "manifest.json").read_bytes(),
                "Archived and standalone manifests differ")
        expected_members = {f"{name}/manifest.json"}
        for entry in manifest["files"]:
            path = PurePosixPath(entry["path"])
            require(not path.is_absolute() and ".." not in path.parts,
                    f"Unexpected manifest path: {path}")
            member = f"{name}/{path}"
            require(member not in expected_members, f"Duplicate manifest path: {path}")
            expected_members.add(member)
            file = package / path
            require(file.stat().st_size == entry["bytes"] and sha(file) == entry["sha256"],
                    f"Archived file verification failed: {path}")
        require(member_names == expected_members, "Archive inventory differs from its manifest")
        executable = package / "calc"
        require(sha(executable) == manifest["binary_sha256"] == sha(dist / "calc"),
                "Archived and standalone binaries differ")
        require(executable.stat().st_mode & stat.S_IXUSR, "Archived binary is not executable")

        # A separate short runtime path stays within Linux's Unix-socket limit.
        with tempfile.TemporaryDirectory(prefix="omni-ci-") as runtime:
            environment = os.environ.copy()
            for key in ("OMNI_MODEL", "OMNI_BACKEND", "OMNI_THREADS"):
                environment.pop(key, None)
            environment["OMNI_RUNTIME_DIR"] = runtime
            requests = [
                ("auto", "wat iz 1 + 1", 2, "integer"),
                ("auto", "add the squares of 6 and 8", 100, "integer"),
                ("needle", "convert 2.75 km to m", "2750 m", "quantity"),
            ]
            for backend, request, expected, kind in requests:
                process = subprocess.run(
                    [str(executable), "--no-daemon", "--json", "--retry", "0",
                     "--timeout-ms", "15000", "--threads", "2", "--backend", backend, request],
                    env=environment, capture_output=True, text=True, timeout=20, check=False,
                )
                require(process.returncode == 0 and not process.stderr,
                        f"Packaged CLI failed for {request!r}: {process.stdout} {process.stderr}")
                response = json.loads(process.stdout)
                require(response.get("ok") is True
                        and response.get("result", {}).get("value") == expected
                        and response["result"]["kind"] == kind,
                        f"Packaged CLI returned a wrong answer for {request!r}: {response}")
                if backend == "needle":
                    require(response["model"].startswith("needle"), "Forced Needle request used another model")

    print(json.dumps({"verified_files": len(manifest["files"]),
                      "source_revision": manifest["source_revision"],
                      "binary_sha256": manifest["binary_sha256"],
                      "packaged_cli_checks": len(requests)}, indent=2))


if __name__ == "__main__":
    main()
