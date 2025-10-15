"""
系統資訊收集模組，負責偵測 OS/CPU/GPU/記憶體 與 Python 版本等資料。
所有回傳均以結構化資料表示，供 CLI 與其他模組重複使用。
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

try:
    import psutil  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - 避免在未安裝 psutil 時失敗
    psutil = None

try:
    import GPUtil  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    GPUtil = None


@dataclass
class CPUInfo:
    """儲存 CPU 相關資訊。"""

    physical_cores: Optional[int]
    logical_cores: Optional[int]
    architecture: str
    model: Optional[str]


@dataclass
class GPUInfo:
    """儲存 GPU 相關資訊。"""

    name: str
    memory_total_gb: Optional[float]
    driver_version: Optional[str]
    cuda_version: Optional[str]


@dataclass
class MemoryInfo:
    """儲存記憶體相關資訊。"""

    total_gb: Optional[float]
    available_gb: Optional[float]


@dataclass
class SystemInfo:
    """封裝整體系統資訊。"""

    os_name: str
    os_version: str
    python_version: str
    cpu: CPUInfo
    gpus: List[GPUInfo]
    memory: MemoryInfo

    def as_dict(self) -> Dict[str, Any]:
        """轉換為字典，方便序列化輸出。"""

        return asdict(self)


def _run_command(command: List[str]) -> subprocess.CompletedProcess[str]:
    """安全執行指令並回傳 CompletedProcess。"""

    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )


def _detect_os() -> Dict[str, str]:
    """偵測作業系統名稱與版本。"""

    os_name = platform.system() or "Unknown"
    if os_name == "Linux":
        os_version = platform.release()
    elif os_name == "Windows":
        os_version = platform.version()
    elif os_name == "Darwin":
        os_name = "macOS"
        os_version = platform.mac_ver()[0]
    else:  # pragma: no cover - 其他 OS 皆為未知
        os_version = "Unknown"
    return {"os_name": os_name, "os_version": os_version}


def _detect_cpu() -> CPUInfo:
    """偵測 CPU 型號與核心資訊。"""

    physical = None
    logical = os.cpu_count()

    model = platform.processor() or None
    architecture = platform.machine() or "Unknown"

    if psutil is not None:
        try:
            physical = psutil.cpu_count(logical=False)
            logical = psutil.cpu_count(logical=True)
        except Exception:  # pragma: no cover - 容錯處理
            pass
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq and model:
                # 將主頻資訊補充到型號字串中，方便呈現
                model = f"{model} @ {cpu_freq.current / 1000:.2f}GHz"
        except Exception:  # pragma: no cover
            pass

    return CPUInfo(
        physical_cores=physical,
        logical_cores=logical,
        architecture=architecture,
        model=model,
    )


def _parse_cuda_version(nvidia_smi_output: str) -> Optional[str]:
    """從 `nvidia-smi` 輸出中解析 CUDA 版本。"""

    match = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi_output)
    return match.group(1) if match else None


def _detect_gpus_with_nvidia_smi() -> List[GPUInfo]:
    """透過 `nvidia-smi` 探測 NVIDIA GPU。"""

    result_full = _run_command(["nvidia-smi"])
    if result_full.returncode != 0:
        return []

    cuda_version = _parse_cuda_version(result_full.stdout)
    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return []

    gpus: List[GPUInfo] = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts:
            continue

        name = parts[0]
        memory_total = None
        driver_version = None
        if len(parts) >= 2:
            try:
                memory_total = float(parts[1]) / 1024.0  # 轉換成 GiB
            except ValueError:
                memory_total = None
        if len(parts) >= 3:
            driver_version = parts[2]

        gpus.append(
            GPUInfo(
                name=name,
                memory_total_gb=memory_total,
                driver_version=driver_version,
                cuda_version=cuda_version,
            )
        )

    return gpus


def _detect_gpus_with_gputil() -> List[GPUInfo]:
    """透過 GPUtil 探測 GPU；作為 `nvidia-smi` 的回退方案。"""

    if GPUtil is None:
        return []

    gpus: List[GPUInfo] = []
    try:
        for gpu in GPUtil.getGPUs():
            gpus.append(
                GPUInfo(
                    name=gpu.name,
                    memory_total_gb=getattr(gpu, "memoryTotal", None),
                    driver_version=None,
                    cuda_version=None,
                )
            )
    except Exception:  # pragma: no cover - 防止外部套件異常
        return []
    return gpus


def _detect_gpus() -> List[GPUInfo]:
    """整合多種手段偵測 GPU。"""

    gpus = _detect_gpus_with_nvidia_smi()
    if gpus:
        return gpus
    return _detect_gpus_with_gputil()


def _detect_memory() -> MemoryInfo:
    """偵測記憶體總量與可用量。"""

    total = None
    available = None

    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            total = vm.total / (1024**3)
            available = vm.available / (1024**3)
        except Exception:  # pragma: no cover
            pass

    if total is None:
        try:
            if hasattr(os, "sysconf"):
                page_size = os.sysconf("SC_PAGE_SIZE")  # type: ignore[arg-type]
                phys_pages = os.sysconf("SC_PHYS_PAGES")  # type: ignore[arg-type]
                total = page_size * phys_pages / (1024**3)
        except (ValueError, OSError, AttributeError):  # pragma: no cover
            pass

    return MemoryInfo(total_gb=total, available_gb=available)


def collect_system_info() -> SystemInfo:
    """主函式：收集並彙整整體系統資訊。"""

    os_info = _detect_os()
    cpu_info = _detect_cpu()
    gpu_info = _detect_gpus()
    memory_info = _detect_memory()

    return SystemInfo(
        os_name=os_info["os_name"],
        os_version=os_info["os_version"],
        python_version=sys.version.split()[0],
        cpu=cpu_info,
        gpus=gpu_info,
        memory=memory_info,
    )


def collect_system_info_as_dict() -> Dict[str, Any]:
    """回傳字典格式，方便序列化或轉交其他模組。"""

    return collect_system_info().as_dict()


__all__ = [
    "CPUInfo",
    "GPUInfo",
    "MemoryInfo",
    "SystemInfo",
    "collect_system_info",
    "collect_system_info_as_dict",
]

