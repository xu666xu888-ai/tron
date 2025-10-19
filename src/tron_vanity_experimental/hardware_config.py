# -*- coding: utf-8 -*-
"""
硬體自適應配置模組：偵測當前環境（GPU/CPU），提供最佳化批次、進程與執行緒設定。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional, Tuple

try:
    import cupy as cp  # type: ignore
except ImportError:  # pragma: no cover - 無 GPU/CuPy 時退化為 CPU 模式
    cp = None  # type: ignore

from .system_info import collect_system_info

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareAdaptiveConfig:
    """彙整 GPU/CPU 環境下需要的參數設定。"""

    backend: str  # "GPU" 或 "CPU"
    profile: str
    name: str
    sm_count: Optional[int]
    total_mem_gb: float
    compute_capability: Optional[Tuple[int, int]]
    default_batches: Tuple[int, ...]
    default_streams: int
    wnaf_threshold: int
    secp_threads: int
    keccak_threads: int
    sha_threads: int
    base58_threads: int
    max_pending_multiplier: int
    memory_pool_limit_bytes: Optional[int]
    cpu_physical_cores: Optional[int]
    cpu_logical_cores: Optional[int]
    recommended_processes: int
    max_batch_size: int
    estimated_addr_per_sec: int
    cupy_available: bool
    notes: Tuple[str, ...] = ()
    gpu_count: int = 0
    device_ids: Tuple[int, ...] = ()
    device_names: Tuple[str, ...] = ()
    total_vram_gb_all: float = 0.0
    aggregate_batch_hint: Tuple[int, ...] = ()

    def summary(self) -> str:
        """提供讀取友善的摘要字串。"""

        if self.backend == "GPU" and self.compute_capability:
            major, minor = self.compute_capability
            gpu_suffix = f" x{self.gpu_count}" if self.gpu_count > 1 else ""
            return (
                f"[GPU] {self.profile} ({self.name}{gpu_suffix}, SM={self.sm_count}, "
                f"CC={major}.{minor}, VRAM={self.total_mem_gb:.1f} GiB)"
            )
        return (
            f"[CPU] {self.profile} ({self.name}, "
            f"RAM={self.total_mem_gb:.1f} GiB, Cores={self.recommended_processes})"
        )


def _align(value: int, alignment: int = 256) -> int:
    """將數值對齊到指定倍數。"""

    if value <= 0:
        return alignment
    return ((value + alignment - 1) // alignment) * alignment


def _profile_from_props(name: str, sm_count: int, major: int, total_mem_gb: float) -> str:
    """依據 GPU 名稱與規格推定最佳化 Profile。"""

    upper = name.upper()
    if "H100" in upper or (major >= 9 and sm_count >= 120):
        return "H100"
    if "A100" in upper or (major == 8 and sm_count >= 108 and total_mem_gb >= 40):
        return "A100"
    if "L40S" in upper or "L40" in upper:
        return "L40S"
    if "4090" in upper or ("RTX" in upper and "4090" in upper):
        return "RTX4090"
    if "L4" in upper:
        return "L4"
    if major >= 8 and sm_count >= 120:
        return "Ada-Large"
    if major >= 7 and sm_count >= 80:
        return "Ampere-Large"
    return "Generic"


def _profile_template(profile: str, name: str, sm_count: int, total_mem_gb: float,
                      major: int, minor: int) -> HardwareAdaptiveConfig:
    """回傳對應 GPU profile 的預設參數模板。"""

    base = HardwareAdaptiveConfig(
        backend="GPU",
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
        cpu_physical_cores=None,
        cpu_logical_cores=None,
        recommended_processes=1,
        max_batch_size=262144,
        estimated_addr_per_sec=300_000,
        cupy_available=True,
    )

    if profile == "H100":
        return replace(
            base,
            default_batches=(131072, 262144, 393216, 524288, 786432, 1048576, 1310720),
            default_streams=8,
            wnaf_threshold=524288,
            secp_threads=512,
            keccak_threads=512,
            sha_threads=512,
            base58_threads=512,
            max_pending_multiplier=4,
            memory_pool_limit_bytes=int(total_mem_gb * (1024**3) * 0.35),
            estimated_addr_per_sec=2_000_000,
        )
    if profile == "A100":
        return replace(
            base,
            default_batches=(65536, 131072, 262144, 393216, 524288, 786432, 983040),
            default_streams=8,
            wnaf_threshold=262144,
            secp_threads=512,
            keccak_threads=512,
            sha_threads=512,
            base58_threads=512,
            max_pending_multiplier=4,
            memory_pool_limit_bytes=int(total_mem_gb * (1024**3) * 0.30),
            estimated_addr_per_sec=1_200_000,
        )
    if profile == "L40S":
        return replace(
            base,
            default_batches=(131072, 262144, 393216, 524288, 786432, 1048576),
            default_streams=12,
            wnaf_threshold=393216,
            secp_threads=512,
            keccak_threads=512,
            sha_threads=512,
            base58_threads=512,
            max_pending_multiplier=5,
            memory_pool_limit_bytes=int(total_mem_gb * (1024**3) * 0.30),
            estimated_addr_per_sec=1_600_000,
        )
    if profile in {"RTX4090", "Ada-Large"}:
        return replace(
            base,
            default_batches=(32768, 65536, 131072, 262144, 393216, 524288),
            default_streams=8,
            wnaf_threshold=262144,
            secp_threads=384,
            keccak_threads=384,
            sha_threads=384,
            base58_threads=384,
            max_pending_multiplier=4,
            memory_pool_limit_bytes=int(total_mem_gb * (1024**3) * 0.25),
            estimated_addr_per_sec=900_000,
        )
    if profile == "L4":
        return replace(
            base,
            default_batches=(
                262144,
                393216,
                524288,
                786432,
                1048576,
                1310720,
                1572864,
                2097152,
                2621440,
                3145728,
            ),
            default_streams=16,
            wnaf_threshold=262144,
            secp_threads=512,
            keccak_threads=512,
            sha_threads=512,
            base58_threads=512,
            max_pending_multiplier=6,
            memory_pool_limit_bytes=int(total_mem_gb * (1024**3) * 0.35),
            max_batch_size=3145728,
            estimated_addr_per_sec=1_100_000,
        )
    if profile == "Ampere-Large":
        return replace(
            base,
            default_batches=(32768, 65536, 131072, 196608, 262144, 327680, 393216),
            default_streams=6,
            wnaf_threshold=196608,
            secp_threads=384,
            keccak_threads=384,
            sha_threads=384,
            base58_threads=384,
            max_pending_multiplier=3,
            memory_pool_limit_bytes=int(total_mem_gb * (1024**3) * 0.25),
            estimated_addr_per_sec=600_000,
        )
    return base


def _suggest_processes(physical: Optional[int], logical: Optional[int]) -> int:
    """依 CPU 規格推估建議的進程數。"""

    if physical and physical > 0:
        return max(1, min(physical, 32))
    if logical and logical > 0:
        return max(1, min(logical - 1, 32))
    return 1


def _compute_max_batch(defaults: Tuple[int, ...], total_mem_gb: float, backend: str) -> int:
    """依據記憶體容量（GPU/CPU）調整批次上限。"""

    if total_mem_gb <= 0:
        total_mem_gb = 1.0
    if backend == "GPU":
        factor = 200_000  # 約 0.2M addr / GiB，L4(15.6 GiB) -> ~3.1M
    else:
        factor = 1_024  # 16 GiB -> 約 16k
    mem_limit = _align(int(total_mem_gb * factor))
    base_max = max(defaults) if defaults else mem_limit
    return max(256, min(mem_limit, base_max))


def _apply_cpu_context(config: HardwareAdaptiveConfig, cupy_ok: bool) -> HardwareAdaptiveConfig:
    """補齊 CPU 資訊與批次限制。"""

    sysinfo = collect_system_info()
    recommended = _suggest_processes(sysinfo.cpu.physical_cores, sysinfo.cpu.logical_cores)
    total_mem_gb = sysinfo.memory.total_gb or config.total_mem_gb or 8.0

    max_batch = _compute_max_batch(config.default_batches, total_mem_gb, config.backend)
    filtered_batches = tuple(b for b in config.default_batches if b <= max_batch)
    if not filtered_batches:
        filtered_batches = (max_batch,)

    notes = list(config.notes)
    if config.backend == "CPU" and sysinfo.gpus:
        notes.append("偵測到 NVIDIA GPU，但缺少 CuPy 或 CUDA 驅動，已退回 CPU 模式")

    return replace(
        config,
        cpu_physical_cores=sysinfo.cpu.physical_cores,
        cpu_logical_cores=sysinfo.cpu.logical_cores,
        recommended_processes=recommended,
        total_mem_gb=total_mem_gb,
        default_batches=filtered_batches,
        max_batch_size=max_batch,
        cupy_available=cupy_ok,
        notes=tuple(notes),
    )


def _build_gpu_config() -> HardwareAdaptiveConfig:
    """建立 GPU 模式配置。"""

    if cp is None:
        raise RuntimeError("CuPy 尚未安裝，無法建立 GPU 配置")

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count <= 0:
            raise RuntimeError("沒有可用的 CUDA 裝置")
        device_infos = []
        best_idx = 0
        best_score = -1
        for dev_id in range(device_count):
            props = cp.cuda.runtime.getDeviceProperties(dev_id)
            name = props["name"].decode()
            sm_count = int(props["multiProcessorCount"])
            total_mem = int(props["totalGlobalMem"])
            total_mem_gb = total_mem / (1024**3)
            major = int(props["major"])
            minor = int(props["minor"])
            clock = int(props.get("clockRate", 0))
            score = sm_count * max(clock, 1)
            if score > best_score:
                best_idx = dev_id
                best_score = score
            device_infos.append(
                {
                    "id": dev_id,
                    "name": name,
                    "sm_count": sm_count,
                    "total_mem_gb": total_mem_gb,
                    "major": major,
                    "minor": minor,
                }
            )
        primary = device_infos[best_idx]
        name = primary["name"]
        sm_count = primary["sm_count"]
        total_mem_gb = primary["total_mem_gb"]
        major = primary["major"]
        minor = primary["minor"]
    except Exception as exc:  # pragma: no cover - 覆蓋驅動異常
        raise RuntimeError("無法透過 CuPy 取得 GPU 屬性") from exc

    profile = _profile_from_props(name, sm_count, major, total_mem_gb)
    template = _profile_template(profile, name, sm_count, total_mem_gb, major, minor)
    config = _apply_cpu_context(template, cupy_ok=True)
    total_vram_all = sum(info["total_mem_gb"] for info in device_infos)
    aggregate_batches = tuple(_align(b * device_count) for b in config.default_batches)
    names = tuple(info["name"] for info in device_infos)
    device_ids = tuple(info["id"] for info in device_infos)

    notes = list(config.notes)
    if device_count > 1:
        notes.append(
            "Multi-GPU 模式：偵測到 {} 張卡 ({})".format(
                device_count, ", ".join(names)
            )
        )
        notes.append("default_batches 為單卡建議，aggregate_batch_hint 為多卡總量參考")
        pending_mul = max(config.max_pending_multiplier, device_count * 2)
        config = replace(
            config,
            notes=tuple(notes),
            max_pending_multiplier=pending_mul,
            gpu_count=device_count,
            device_ids=device_ids,
            device_names=names,
            total_vram_gb_all=total_vram_all,
            aggregate_batch_hint=aggregate_batches,
        )
    else:
        config = replace(
            config,
            notes=tuple(notes),
            gpu_count=1,
            device_ids=device_ids,
            device_names=names,
            total_vram_gb_all=total_vram_all,
            aggregate_batch_hint=config.default_batches,
        )

    _LOGGER.info("偵測 GPU：%s", config.summary())
    return config


def _build_cpu_config() -> HardwareAdaptiveConfig:
    """建立 CPU 模式配置（無 GPU 或缺少 CuPy）。"""

    sysinfo = collect_system_info()
    total_mem = sysinfo.memory.total_gb or 8.0
    recommended = _suggest_processes(sysinfo.cpu.physical_cores, sysinfo.cpu.logical_cores)
    default_batches = (1024, 2048, 4096, 8192, 16384)

    config = HardwareAdaptiveConfig(
        backend="CPU",
        profile="CPU",
        name=sysinfo.cpu.model or "Generic CPU",
        sm_count=None,
        total_mem_gb=total_mem,
        compute_capability=None,
        default_batches=default_batches,
        default_streams=0,
        wnaf_threshold=0,
        secp_threads=0,
        keccak_threads=0,
        sha_threads=0,
        base58_threads=0,
        max_pending_multiplier=max(1, recommended // 2) if recommended > 1 else 1,
        memory_pool_limit_bytes=None,
        cpu_physical_cores=sysinfo.cpu.physical_cores,
        cpu_logical_cores=sysinfo.cpu.logical_cores,
        recommended_processes=recommended,
        max_batch_size=0,  # 後續由 _apply_cpu_context 更新
        estimated_addr_per_sec=max(10_000, recommended * 5_000),
        cupy_available=False,
        notes=(),
    )
    config = _apply_cpu_context(config, cupy_ok=False)
    notes = list(config.notes)
    notes.append("目前以 CPU 模式運行，建議安裝 CuPy 以啟用 GPU 加速")
    config = replace(config, notes=tuple(notes))
    _LOGGER.warning("未偵測到可用 GPU，使用 CPU 配置：%s", config.summary())
    return config


def detect_hardware_config(force_cpu: bool = False) -> HardwareAdaptiveConfig:
    """對外介面：回傳當前最佳化硬體配置，可選擇強制 CPU 模式。"""

    if not force_cpu and cp is not None:
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                return _build_gpu_config()
        except Exception:  # pragma: no cover - 沒有 GPU 或驅動未載入
            pass
    return _build_cpu_config()


HARDWARE_CONFIG = detect_hardware_config()


def get_hardware_config() -> HardwareAdaptiveConfig:
    """提供外部模組存取硬體配置。"""

    return HARDWARE_CONFIG


__all__ = [
    "HardwareAdaptiveConfig",
    "HARDWARE_CONFIG",
    "get_hardware_config",
]
