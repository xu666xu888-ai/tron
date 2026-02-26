# -*- coding: utf-8 -*-
"""
環境檢查腳本：
- Python/OS 資訊
- CUDA/NVIDIA（nvidia-smi / nvcc）
- 必要套件版本與可用性
- TronGrid API Key 檢查（若需呼叫 validateaddress）
"""
from __future__ import annotations

import os
import sys
import platform
import shutil
import subprocess


PKGS = [
    ("coincurve", "__version__"),
    ("base58", "__version__"),
    ("tronpy", "__version__"),
    ("requests", "__version__"),
    ("sha3", None),  # pycryptodome 提供 Crypto.Hash.keccak，此處檢測舊版相容
]

# 額外檢查 pycryptodome
EXTRA_PKGS = [
    ("Crypto.Hash.keccak", None),
]


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
        return 0, out.strip()
    except Exception as e:
        return 1, str(e)


def main() -> int:
    print("== Python & OS ==")
    print(sys.version)
    print(platform.platform())

    print("\n== NVIDIA/CUDA ==")
    if shutil.which("nvidia-smi"):
        code, out = run(["nvidia-smi"])
        print(out.splitlines()[0] if out else "nvidia-smi ok")
    else:
        print("nvidia-smi not found")

    if shutil.which("nvcc"):
        code, out = run(["nvcc", "--version"])
        print(out.splitlines()[-1] if out else "nvcc ok")
    else:
        print("nvcc not found")

    print("\n== Python Packages ==")
    for mod, ver_attr in PKGS:
        try:
            m = __import__(mod)
            v = getattr(m, ver_attr) if ver_attr else "OK"
            print(f"{mod}: {v}")
        except Exception as e:
            print(f"{mod}: MISSING ({e})")

    # CuPy（可選）
    try:
        import cupy as cp  # noqa: F401
        print("cupy: OK")
    except Exception as e:
        print(f"cupy: MISSING/OPTIONAL ({e})")

    print("\n== TronGrid config ==")
    api_key = os.environ.get("TRON_PRO_API_KEY")
    url = os.environ.get("TRON_GRID_URL", "https://api.trongrid.io")
    print("TRON_GRID_URL:", url)
    print("TRON_PRO_API_KEY set:", bool(api_key))

    print("\n完成環境檢查。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
