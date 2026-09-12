#!/usr/bin/env python3
"""Exercise the fused GPU pipeline without exposing generated keys."""

from __future__ import annotations

import secrets

import cupy as cp
from tron_vanity.addr import is_valid_tron_base58, privkey_to_tron_address
from tron_vanity.gpu_incremental import BATCH_K, gpu_vanity_scan
from tronpy.keys import PrivateKey


def main() -> int:
    suffix = "1"
    total_threads = 256
    hits, scanned = gpu_vanity_scan(
        secrets.token_bytes(32),
        suffix=suffix,
        total_threads=total_threads,
        iters_per_thread=1,
    )
    if scanned != total_threads * BATCH_K:
        raise RuntimeError("GPU 掃描數量與配置不一致")
    if not hits:
        raise RuntimeError("短尾號驗證批次沒有命中，無法交叉檢查")

    for private_key, address_hex, address_base58 in hits:
        local_hex, local_base58 = privkey_to_tron_address(private_key)
        independent_base58 = PrivateKey(private_key).public_key.to_base58check_address()
        if not (
            local_hex == address_hex
            and local_base58 == address_base58
            and independent_base58 == address_base58
            and is_valid_tron_base58(address_base58)
            and address_base58.endswith(suffix)
        ):
            raise RuntimeError("GPU 命中未通過 CPU 與 tronpy 交叉驗證")

    device = cp.cuda.Device(0)
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"].decode() if isinstance(properties["name"], bytes) else properties["name"]
    print(f"GPU_DEVICE={name}")
    print(f"GPU_PIPELINE_SCANNED={scanned}")
    print(f"GPU_PIPELINE_HITS={len(hits)}")
    print("GPU_PIPELINE_ALL_HITS_VERIFIED=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
