#!/usr/bin/env python3
"""Collect exact available license/notice files from the offline Cargo closure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TARGET = "x86_64-unknown-linux-gnu"
NAMES = re.compile(r"^(?:unlicen[cs]e|licen[cs]e|copying|notice|authors|copyright)(?:[._-].*)?$", re.IGNORECASE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    command = ["cargo", "metadata", "--locked", "--offline", "--format-version", "1", "--filter-platform", TARGET]
    metadata = json.loads(subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout)
    packages = {package["id"]: package for package in metadata["packages"]}
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    root_id = metadata["resolve"]["root"]
    selected, queue = set(), [root_id]
    while queue:
        current = queue.pop()
        if current in selected:
            continue
        selected.add(current)
        for dep in nodes[current]["deps"]:
            if any(kind["kind"] != "dev" for kind in dep["dep_kinds"]):
                queue.append(dep["pkg"])
    selected.remove(root_id)
    entries = []
    for package in sorted((packages[key] for key in selected), key=lambda p: (p["name"],p["version"])):
        source_dir = Path(package["manifest_path"]).parent
        destination = HERE / f'{package["name"]}-{package["version"]}'
        destination.mkdir(exist_ok=True)
        candidates = set()
        for source in source_dir.rglob("*"):
            if not source.is_file() or source.is_symlink():
                continue
            relative = source.relative_to(source_dir)
            if any(part in {"target", ".git"} for part in relative.parts):
                continue
            if NAMES.match(source.name):
                candidates.add(source)
        if package.get("license_file"):
            declared = source_dir / package["license_file"]
            if declared.is_file():
                candidates.add(declared)
        if package["name"] == "fend-core" and (source_dir/"OMNI-PATCH.md").is_file():
            candidates.add(source_dir/"OMNI-PATCH.md")
        files = []
        for source in sorted(candidates):
            relative = source.relative_to(source_dir)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            assert sha(source) == sha(target)
            files.append({"path": str(target.relative_to(HERE)), "source_path": str(relative), "sha256": sha(target), "bytes": target.stat().st_size,
                          "role":"notice" if source.name.lower().startswith(("notice","authors","copyright","omni-patch")) else "license"})
        if source_dir.is_relative_to(ROOT):
            source = "path:" + str(source_dir.relative_to(ROOT))
        else:
            source = package["source"]
        entry = {
            "name":package["name"], "version":package["version"],
            "license":package.get("license"), "declared_license_file":package.get("license_file"),
            "source":source, "repository":package.get("repository"),
            "crate_url":f'https://crates.io/crates/{package["name"]}/{package["version"]}',
            "files":files,
            "status":"copied" if files and (package.get("license") or package.get("license_file")) else "unresolved",
        }
        if not files:
            entry["issue"]="No license/notice text is present in the cached published source or vendored source. License metadata alone is not the full text."
        elif not package.get("license") and not package.get("license_file"):
            entry["issue"]="Missing Cargo license metadata; copied available notices without inferring a license."
        entries.append(entry)
    sysroot = Path(subprocess.run(["rustc","--print","sysroot"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip())
    rustc_version = subprocess.run(["rustc","-vV"],cwd=ROOT,check=True,capture_output=True,text=True).stdout
    rustc_fields = dict(line.split(": ",1) for line in rustc_version.splitlines() if ": " in line)
    standard_library = {"name":"Rust Standard Library", "version":rustc_fields["release"],
                        "license":"MIT OR Apache-2.0, with per-file and dependency notices in the copied document",
                        "source":f'https://github.com/rust-lang/rust/tree/{rustc_fields["commit-hash"]}/library',
                        "rustc":rustc_fields,"files":[],"status":"unresolved"}
    std_notice = sysroot/"share/doc/rust/COPYRIGHT-library.html"
    if std_notice.is_file():
        target = HERE/f'rust-std-{rustc_fields["release"]}'/std_notice.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(std_notice,target)
        assert sha(std_notice)==sha(target)
        standard_library["files"].append({"path":str(target.relative_to(HERE)),"source_path":"share/doc/rust/COPYRIGHT-library.html","sha256":sha(target),"bytes":target.stat().st_size,"role":"copyright and full license notices"})
        standard_library["status"]="copied"
    else:
        standard_library["issue"]="The installed toolchain does not contain the Rust Standard Library copyright/license document."
    report = {
        "target":TARGET,
        "scope":"Default-feature normal and build dependency closure for the Linux target, including proc macros and their build dependencies. Dev-only edges and the application's own package are excluded. This records available texts and metadata; it does not choose among dual-license alternatives.",
        "cargo_command":command,
        "cargo_lock_sha256":sha(ROOT/"Cargo.lock"),
        "cargo_toml_sha256":sha(ROOT/"Cargo.toml"),
        "root_features":nodes[root_id]["features"],
        "crate_count":len(entries),
        "file_count":sum(len(entry["files"]) for entry in entries),
        "unresolved":[{"crate":f'{entry["name"]}-{entry["version"]}',"issue":entry["issue"]} for entry in entries if entry["status"]=="unresolved"],
        "crates":entries,
        "runtime_components":[standard_library],
        "text_availability_notes":[{"crate":"wide-0.7.33","note":"Cargo metadata lists Zlib OR Apache-2.0 OR MIT, but this cached published crate includes only LICENSE-ZLIB.md. Its complete Zlib text is copied unchanged; missing alternate texts were not fabricated and no license alternative is selected here."}],
    }
    if standard_library["status"]=="unresolved":
        report["unresolved"].append({"component":"Rust Standard Library","issue":standard_library["issue"]})
    report["file_count"]+=sum(len(component["files"]) for component in report["runtime_components"])
    (HERE/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    checksums = sorted((entry["path"],entry["sha256"]) for component in entries+report["runtime_components"] for entry in component["files"])
    (HERE/"SHA256SUMS").write_text("".join(f'{checksum}  {path}\n' for path,checksum in checksums))
    print(json.dumps({key:report[key] for key in ["root_features","crate_count","file_count","unresolved"]}, indent=2))


if __name__=="__main__":
    main()
