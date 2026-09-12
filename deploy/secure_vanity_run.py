#!/usr/bin/env python3
"""Run GPU search, save the result, and display the verified private key."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

BASE58_SUFFIX = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{1,20}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="執行 TRON GPU 尾號搜尋；命中後顯示私鑰並保存權限 0600 的結果檔。"
    )
    parser.add_argument("suffix", help="Base58 地址尾號，例如 88888")
    parser.add_argument("--timeout", type=float, help="最多執行秒數")
    parser.add_argument("--max-attempts", type=int, help="最多掃描次數")
    return parser.parse_args()


def write_private_result(path: Path, record: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    suffix = args.suffix.strip()
    if not BASE58_SUFFIX.fullmatch(suffix):
        raise SystemExit("尾號必須是 1-20 個 Base58 字元（不含 0、O、I、l）。")
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        raise SystemExit("--timeout 必須大於 0。")
    if args.max_attempts is not None and args.max_attempts <= 0:
        raise SystemExit("--max-attempts 必須大於 0。")

    from tron_vanity.addr import is_valid_tron_base58, privkey_to_tron_address
    from tron_vanity.turbo_search import search_vanity_turbo
    from tronpy.keys import PrivateKey

    os.umask(0o077)
    results_dir = Path.home() / "tron-results"
    results_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(results_dir, 0o700)

    print(f"開始搜尋尾號 {suffix}；命中後將顯示私鑰並保存結果檔。", flush=True)
    last_report = 0.0

    def progress(total_keys: int, elapsed: float, mkeys_per_sec: float) -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report >= 5:
            print(
                f"進度：{elapsed:.1f}s，{mkeys_per_sec:.2f} Mkeys/s，"
                f"已掃描 {total_keys:,} 組",
                flush=True,
            )
            last_report = now

    result = search_vanity_turbo(
        suffix=suffix,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        gpu_ids=[0],
        progress_callback=progress,
    )
    if not result.get("found"):
        print(
            f"未命中：{result.get('reason', 'unknown')}；"
            f"已掃描 {int(result.get('total_keys', 0)):,} 組。",
            flush=True,
        )
        return 1

    private_key = bytes.fromhex(result["privkey_hex"])
    local_hex, local_base58 = privkey_to_tron_address(private_key)
    independent_base58 = PrivateKey(private_key).public_key.to_base58check_address()
    verified = (
        local_hex == result["address_hex"]
        and local_base58 == result["address_base58"]
        and independent_base58 == result["address_base58"]
        and is_valid_tron_base58(result["address_base58"])
        and result["address_base58"].endswith(suffix)
    )
    if not verified:
        raise RuntimeError("GPU 命中結果未通過 CPU 與 tronpy 交叉驗證；未寫入結果。")

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output_path = results_dir / f"tron-{suffix}-{timestamp}-{uuid.uuid4().hex[:12]}.json"
    commit_file = Path(__file__).resolve().parent / ".deployed-commit"
    record = {
        "suffix": suffix,
        "address_base58": result["address_base58"],
        "address_hex": result["address_hex"],
        "privkey_hex": result["privkey_hex"],
        "verified_local": True,
        "verified_tronpy": True,
        "attempts": int(result["total_keys"]),
        "elapsed_seconds": float(result["elapsed"]),
        "mkeys_per_second": float(result["mkeys_per_sec"]),
        "gpu_id": int(result["gpu_id"]),
        "repo_commit": commit_file.read_text().strip() if commit_file.exists() else "unknown",
        "created_at_utc": timestamp,
    }
    write_private_result(output_path, record)

    print(f"命中地址：{result['address_base58']}", flush=True)
    print(f"私鑰 (HEX)：{result['privkey_hex']}", flush=True)
    print("CPU 與 tronpy 交叉驗證：通過", flush=True)
    print(
        f"效能：{result['mkeys_per_sec']:.2f} Mkeys/s；"
        f"耗時 {result['elapsed']:.2f}s；掃描 {int(result['total_keys']):,} 組",
        flush=True,
    )
    print(f"私密結果檔：{output_path}（權限 0600）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
