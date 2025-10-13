# -*- coding: utf-8 -*-
"""
根據當前 GPU 自動推導最佳化參數的硬體配置模組。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

try:
    import cupy as cp  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("需要安裝 CuPy 才能使用硬體自適應配置：pip install cupy-cuda11x/12x") from exc


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareAdaptiveConfig:
    profile: str
    name: str
    sm_count: int
    total_mem_gb: float
    compute_capability: Tuple[int, int]
    default_batches: Tuple[int, ...]
    default_streams: int
    wnaf_threshold: int
    secp_threads: int
    keccak_threads: int
    sha_threads: int
    base58_threads: int
    max_pending_multiplier: int
    memory_pool_limit_bytes: int | None = None

    def summary(self) -> str:
        major, minor = self.compute_capability
        return (
            f"{self.profile} ({self.name}, SM={self.sm_count}, "
            f"CC={major}.{minor}, VRAM={self.total_mem_gb:.1f} GiB)"
        )


def _profile_from_props(name: str, sm_count: int, major: int, total_mem_gb: float) -> str:
    upper = name.upper()
    if "H100" in upper or (major >= 9 and sm_count >= 120):
        return "H100"
    if "A100" in upper or (major == 8 and sm_count >= 108 and total_mem_gb >= 40):
        return "A100"
    if "4090" in upper or ("RTX" in upper and "4090" in upper):
        return "RTX4090"
    if "L4" in upper:
        return "L4"
    if major >= 8 and sm_count >= 120:
        return "Ada-Large"
    if major >= 7 and sm_count >= 80:
        return "Ampere-Large"
    return "Generic"


def _config_for_profile(profile: str, name: str, sm_count: int, total_mem_gb: float,
                        major: int, minor: int) -> HardwareAdaptiveConfig:
    if profile == "H100":
        return HardwareAdaptiveConfig(
            profile=profile,
            name=name,
            sm_count=sm_count,
            total_mem_gb=total_mem_gb,
            compute_capability=(major, minor),
            default_batches=(131072, 262144, 393216, 524288, 786432, 1048576, 1310720),
            default_streams=8,
            wnaf_threshold=524288,
            secp_threads=512,
            keccak_threads=512,
            sha_threads=512,
            base58_threads=512,
            max_pending_multiplier=4,
            memory_pool_limit_bytes=int(total_mem_gb * (1024**3) * 0.35),
        )
    if profile == "A100":
        return HardwareAdaptiveConfig(
            profile=profile,
            name=name,
            sm_count=sm_count,
            total_mem_gb=total_mem_gb,
            compute_capability=(major, minor),
            default_batches=(65536, 131072, 262144, 393216, 524288, 786432, 983040),
            default_streams=8,
            wnaf_threshold=262144,
            secp_threads=512,
            keccak_threads=512,
            sha_threads=512,
            base58_threads=512,
            max_pending_multiplier=4,
            memory_pool_limit_bytes=int(total_mem_gb * (1024**3) * 0.30),
        )
    if profile in {"RTX4090", "Ada-Large"}:
        return HardwareAdaptiveConfig(
            profile=profile,
            name=name,
            sm_count=sm_count,
            total_mem_gb=total_mem_gb,
            compute_capability=(major, minor),
            default_batches=(32768, 65536, 131072, 262144, 393216, 524288),
            default_streams=8,
            wnaf_threshold=262144,
            secp_threads=384,
            keccak_threads=384,
            sha_threads=384,
            base58_threads=384,
            max_pending_multiplier=4,
            memory_pool_limit_bytes=int(total_mem_gb * (1024**3) * 0.25),
        )
    if profile == "L4":
        return HardwareAdaptiveConfig(
            profile=profile,
            name=name,
            sm_count=sm_count,
            total_mem_gb=total_mem_gb,
            compute_capability=(major, minor),
            default_batches=(16384, 32768, 65536, 98304, 131072, 262144, 393216),
            default_streams=6,
            wnaf_threshold=131072,
            secp_threads=256,
            keccak_threads=256,
            sha_threads=256,
            base58_threads=256,
            max_pending_multiplier=3,
            memory_pool_limit_bytes=int(total_mem_gb * (1024**3) * 0.25),
        )
    if profile == "Ampere-Large":
        return HardwareAdaptiveConfig(
            profile=profile,
            name=name,
            sm_count=sm_count,
            total_mem_gb=total_mem_gb,
            compute_capability=(major, minor),
            default_batches=(32768, 65536, 131072, 262144, 393216, 524288),
            default_streams=6,
            wnaf_threshold=196608,
            secp_threads=384,
            keccak_threads=384,
            sha_threads=384,
            base58_threads=384,
            max_pending_multiplier=3,
            memory_pool_limit_bytes=int(total_mem_gb * (1024**3) * 0.25),
        )
    # Generic fallback
    return HardwareAdaptiveConfig(
        profile=profile,
        name=name,
        sm_count=sm_count,
        total_mem_gb=total_mem_gb,
        compute_capability=(major, minor),
        default_batches=(16384, 32768, 65536, 131072, 196608, 262144),
        default_streams=4,
        wnaf_threshold=131072,
        secp_threads=256,
        keccak_threads=256,
        sha_threads=256,
        base58_threads=256,
        max_pending_multiplier=3,
        memory_pool_limit_bytes=int(total_mem_gb * (1024**3) * 0.20),
    )


def detect_hardware_config() -> HardwareAdaptiveConfig:
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props["name"].decode()
    sm_count = int(props["multiProcessorCount"])
    total_mem = int(props["totalGlobalMem"])
    total_mem_gb = total_mem / (1024**3)
    major = int(props["major"])
    minor = int(props["minor"])

    profile = _profile_from_props(name, sm_count, major, total_mem_gb)
    config = _config_for_profile(profile, name, sm_count, total_mem_gb, major, minor)
    _LOGGER.info("偵測 GPU：%s", config.summary())
    return config


HARDWARE_CONFIG = detect_hardware_config()


def get_hardware_config() -> HardwareAdaptiveConfig:
    """提供外部模組存取硬體配置。"""
    return HARDWARE_CONFIG


__all__ = ["HardwareAdaptiveConfig", "HARDWARE_CONFIG", "get_hardware_config"]
