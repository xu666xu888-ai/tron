# -*- coding: utf-8 -*-
"""
CUDA 亂數批量產生工具：
- 使用 CuPy 的 GPU 隨機數生成功能，快速生成多組 32 bytes 私鑰候選
- 若無 GPU/CuPy，則自動退化為 os.urandom

注意：僅負責「候選私鑰」的批量產生；secp256k1 橢圓曲線計算仍在 CPU 由 libsecp256k1（coincurve）處理。
"""
from __future__ import annotations

import os
import typing as t


def has_cupy() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def generate_gpu_secrets(n: int) -> t.List[bytes]:
    """以 GPU 產生 n 組 32 bytes 亂數（若無 GPU 則退化為 CPU）。"""
    if n <= 0:
        return []

    try:
        import cupy as cp
        # 直接使用 cp.random.bytes 由 GPU 端生成並返回為 host bytes
        blob: bytes = cp.random.bytes(n * 32)
        out = [blob[i * 32 : (i + 1) * 32] for i in range(n)]
        return out
    except Exception:
        # 退化至 CPU 的 os.urandom
        return [os.urandom(32) for _ in range(n)]


__all__ = ["has_cupy", "generate_gpu_secrets"]
