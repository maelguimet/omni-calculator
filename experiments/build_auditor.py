#!/usr/bin/env python3
"""Build a read-only experiment against an existing release application library."""
import hashlib
import argparse
import json
import pathlib
import subprocess

root=pathlib.Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser()
parser.add_argument("--profile",choices=["debug","release"],default="release")
parser.add_argument("--pipeline",action="store_true")
args=parser.parse_args()
deps=root/f"target/{args.profile}/deps"
library=max(deps.glob("libomni_calc-*.rlib"),key=lambda p:p.stat().st_mtime_ns)
serde=max(deps.glob("libserde_json-*.rlib"),key=lambda p:p.stat().st_mtime_ns)
name="pipeline" if args.pipeline else "audit-calls"
source="pipeline.rs" if args.pipeline else "audit_calls.rs"
output=root/f"experiments/{name}"
subprocess.run(["rustc","--edition=2024","-O",str(root/f"experiments/{source}"),
                "--extern",f"omni_calc={library}","--extern",f"serde_json={serde}",
                "-L",f"dependency={deps}","-o",str(output)],check=True)
manifest={"application_library":str(library.relative_to(root)),
          "application_library_sha256":hashlib.sha256(library.read_bytes()).hexdigest(),
          "auditor_sha256":hashlib.sha256(output.read_bytes()).hexdigest()}
(root/f"experiments/{name}-build.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(manifest,indent=2))
