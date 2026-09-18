#!/usr/bin/env python3
"""Package a verified local release with its documentation, licenses and hashes."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/calc")
    parser.add_argument("--report", type=Path, action="append", default=[])
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    executable = binary.read_bytes()
    model_manifest = json.loads((ROOT / "models/manifest.json").read_text())
    for model in model_manifest["models"]:
        weights = ROOT / model["path"]
        if sha(weights) != model["sha256"] or weights.read_bytes() not in executable:
            raise SystemExit(f"Binary does not embed the documented model: {model['model']}")
    runtime = ROOT / ".cache/needle/libneedle.so"
    if sha(runtime) != model_manifest["runtime_sha256"] or runtime.read_bytes() not in executable:
        raise SystemExit("Binary does not embed the documented native runtime")
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"]
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    name = f"omni-calc-{version}-linux-x86_64"
    # Only a fresh task-owned staging directory is removed on exit. No source,
    # training checkpoint, existing installation, or unrelated directory is touched.
    with tempfile.TemporaryDirectory(prefix=".package-", dir=destination) as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        shutil.copyfile(binary, stage / "calc")
        (stage / "calc").chmod(0o755)
        for source in [ROOT / "README.md", ROOT / "LICENSE", ROOT / "CONTRIBUTING.md"]:
            shutil.copyfile(source, stage / source.name)
        for directory in ["docs", "third_party"]:
            shutil.copytree(ROOT / directory, stage / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.py"))
        for relative in ["training/README.md", "training/composition/README.md",
                         "training/intent/README.md", "training/reference-v2/README.md",
                         "training/router/README.md", "vendor/fend-core/OMNI-PATCH.md",
                         "vendor/fend-core/omni-float-conversion.patch"]:
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        shutil.copytree(ROOT / "models", stage / "models",
                        ignore=shutil.ignore_patterns("*.cact"))
        (stage / "reports").mkdir()
        for report in args.report:
            data = json.loads(report.read_text())
            if "binary_sha256" in data and data["binary_sha256"] != sha(binary):
                raise SystemExit(f"Report was made with a different binary: {report}")
            shutil.copyfile(report, stage / "reports" / report.name)
            attachments = [item["log"] for item in data.get("checks", []) if "log" in item]
            if "trace_file" in data:
                attachments.append(data["trace_file"])
            for relative in attachments:
                source = (ROOT / relative).resolve(strict=True)
                if not source.is_relative_to(ROOT / "reports"):
                    raise SystemExit(f"Unexpected report attachment: {relative}")
                shutil.copyfile(source, stage / "reports" / source.name)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True, check=False)
        manifest = {
            "name": "omni-calc", "version": version,
            "source_repository": "https://github.com/maelguimet/omni-calculator",
            "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "platform": "Linux x86_64", "minimum_glibc": "2.39",
            "binary_sha256": sha(binary),
            "embedded_models_verified": True,
            "model_manifest_sha256": sha(ROOT / "models/manifest.json"),
            "files": [{"path": str(file.relative_to(stage)), "bytes": file.stat().st_size,
                       "sha256": sha(file)}
                      for file in sorted(stage.rglob("*")) if file.is_file()],
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Fixed timestamps, ownership and file ordering make repeated packages stable.
        archive = Path(temporary) / (name + ".tar.gz")
        with archive.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as tar:
                    for file in sorted(stage.rglob("*")):
                        if not file.is_file():
                            continue
                        data = file.read_bytes()
                        entry = tarfile.TarInfo(f"{name}/{file.relative_to(stage)}")
                        entry.size = len(data)
                        entry.mode = 0o755 if file.name == "calc" else 0o644
                        entry.mtime = 0
                        tar.addfile(entry, io.BytesIO(data))
        # Verify every archived byte against the staged artifact before publication.
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                source = stage.parent / member.name
                if hashlib.sha256(tar.extractfile(member).read()).hexdigest() != sha(source):
                    raise SystemExit(f"Archive verification failed: {member.name}")
        os.replace(archive, destination / archive.name)
        os.replace(stage / "calc", destination / "calc")
        os.replace(stage / "manifest.json", destination / "manifest.json")
    checksums = [f"{sha(destination / file)}  {file}"
                 for file in ["calc", "manifest.json", name + ".tar.gz"]]
    (destination / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print("\n".join(checksums))


if __name__ == "__main__":
    main()
