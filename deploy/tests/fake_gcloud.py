#!/usr/bin/env python3
"""Offline gcloud contract stub: records commands, never uses the network."""
import json
import os
import sys
import tarfile
from pathlib import Path

args = sys.argv[1:]
root = Path(os.environ["FAKE_CLOUD_DIR"])
mode = os.environ.get("FAKE_CLOUD_MODE", "success")
with (root / "calls.jsonl").open("a") as handle:
    handle.write(json.dumps(args) + "\n")
command = args[:3]
if command == ["compute", "instances", "list"]:
    if mode == "list-denied":
        sys.exit(1)
    if mode == "exists":
        print("test-tron")
elif command == ["compute", "firewall-rules", "list"]:
    if mode == "firewall-denied":
        sys.exit(1)
    print("allow-iap-ssh-tron-vanity")
elif command == ["compute", "instances", "create"]:
    metadata = next(a for a in args if a.startswith("--metadata="))
    token = metadata.split("tron-deploy-token=")[1]
    (root / "token").write_text(token)
    if mode in ("create-ambiguous", "create-foreign"):
        sys.exit(1)
elif command == ["compute", "instances", "describe"]:
    assert not any(a.startswith("--filter=") for a in args), "describe does not support --filter"
    token = "foreign-token" if mode == "create-foreign" else (root / "token").read_text()
    print(json.dumps({"metadata": {"items": [{"key": "tron-deploy-token", "value": token}]}}))
elif command == ["compute", "instances", "delete-access-config"]:
    if mode == "ip-removal-failed":
        sys.exit(1)
elif args[:2] == ["compute", "scp"]:
    with tarfile.open(args[2]) as archive:
        (root / "archive-members.json").write_text(json.dumps(archive.getnames()))
elif args[:2] == ["compute", "ssh"]:
    remote = next(a for a in args if a.startswith("--command="))
    if mode == "install-failed" and "install_app.sh" in remote:
        sys.exit(1)
    if mode == "verify-failed" and "verify_gpu.py" in remote:
        sys.exit(1)
    if mode == "ssh-failed" and remote == "--command=true":
        sys.exit(1)
    if mode == "reboot" and remote == "--command=nvidia-smi >/dev/null":
        marker = root / "rebooted"
        if not marker.exists():
            marker.touch()
            sys.exit(1)
