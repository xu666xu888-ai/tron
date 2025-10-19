"""
依賴檢測模組：負責掃描 Python 套件與系統工具，輸出詳細的安裝狀態報告。
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class PackageStatus:
    """記錄單一 Python 套件的安裝狀態。"""

    display_name: str
    required: bool
    installed: bool
    package_name: Optional[str]
    version: Optional[str]
    detail: Optional[str] = None


@dataclass
class ToolStatus:
    """記錄系統工具的安裝狀態。"""

    name: str
    installed: bool
    version: Optional[str]
    detail: Optional[str] = None


@dataclass
class DependencyReport:
    """整體依賴檢測報告。"""

    python_packages: List[PackageStatus]
    system_tools: List[ToolStatus]

    def all_satisfied(self) -> bool:
        """判斷是否所有必需項目皆已滿足。"""

        packages_ok = all(
            (not pkg.required) or pkg.installed for pkg in self.python_packages
        )
        tools_ok = all(tool.installed for tool in self.system_tools)
        return packages_ok and tools_ok


def _run_command(command: Sequence[str]) -> Tuple[int, str, str]:
    """執行指令並回傳 (returncode, stdout, stderr)。"""

    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"{command[0]} 不存在或無法執行"


def _import_version(package: str) -> Optional[str]:
    """取得套件版本，若未安裝則回傳 None。"""

    try:
        module = importlib.import_module(package)
    except ModuleNotFoundError:
        return None

    version = getattr(module, "__version__", None)
    if version:
        return str(version)

    # 嘗試使用 importlib.metadata 取得版本
    try:  # pragma: no cover - 避免舊版 Python 無此模組
        if sys.version_info >= (3, 8):
            from importlib import metadata
        else:  # pragma: no cover
            import importlib_metadata as metadata  # type: ignore

        return metadata.version(package)
    except Exception:  # pragma: no cover - 此步驟只做補充
        return None


def _check_python_packages() -> List[PackageStatus]:
    """檢查專案所需 Python 套件。"""

    requirements = [
        ("tronpy", True, ("tronpy",)),
        ("ecdsa", True, ("ecdsa",)),
        ("base58", True, ("base58",)),
        ("rich", True, ("rich",)),
        ("psutil", True, ("psutil",)),
        ("GPUtil", False, ("GPUtil",)),
        (
            "CuPy (CUDA 12.x / 11.x)",
            True,
            ("cupy-cuda12x", "cupy-cuda11x", "cupy"),
        ),
    ]

    statuses: List[PackageStatus] = []
    for display_name, required, candidates in requirements:
        installed_name = None
        installed_version = None
        for candidate in candidates:
            installed_version = _import_version(candidate)
            if installed_version is not None:
                installed_name = candidate
                break
        installed = installed_version is not None
        detail = None
        if not installed:
            detail = f"缺少套件：可安裝 {', '.join(candidates)}"
        elif installed_name and installed_name.startswith("cupy-cuda"):
            # 補充當前匹配到的 CUDA 版本
            detail = f"使用 {installed_name}"

        statuses.append(
            PackageStatus(
                display_name=display_name,
                required=required,
                installed=installed,
                package_name=installed_name,
                version=installed_version,
                detail=detail,
            )
        )

    return statuses


def _parse_nvcc_version(stdout: str) -> Optional[str]:
    """解析 `nvcc --version` 的輸出字串。"""

    for line in stdout.splitlines():
        if "release" in line:
            return line.split("release", 1)[-1].strip().strip(",")
    return None


def _check_system_tools() -> List[ToolStatus]:
    """檢查必要的系統工具。"""

    tools: List[ToolStatus] = []

    # 檢查 nvcc（CUDA Toolkit）
    returncode, stdout, stderr = _run_command(["nvcc", "--version"])
    if returncode == 0:
        version = _parse_nvcc_version(stdout)
        tools.append(
            ToolStatus(
                name="CUDA Toolkit (nvcc)",
                installed=True,
                version=version,
            )
        )
    else:
        tools.append(
            ToolStatus(
                name="CUDA Toolkit (nvcc)",
                installed=False,
                version=None,
                detail=stderr or "無法找到 nvcc 指令",
            )
        )

    # 檢查 NVIDIA Driver（透過 nvidia-smi）
    returncode, stdout, stderr = _run_command(["nvidia-smi"])
    if returncode == 0:
        version = None
        for line in stdout.splitlines():
            if "Driver Version" in line:
                version = line.strip()
                break
        tools.append(
            ToolStatus(
                name="NVIDIA Driver (nvidia-smi)",
                installed=True,
                version=version,
            )
        )
    else:
        tools.append(
            ToolStatus(
                name="NVIDIA Driver (nvidia-smi)",
                installed=False,
                version=None,
                detail=stderr or "無法找到 nvidia-smi 指令",
            )
        )

    # 檢查 python3 版本
    returncode, stdout, stderr = _run_command(["python3", "--version"])
    if returncode == 0:
        version = stdout.split()[-1] if stdout else None
        major_minor = tuple(int(x) for x in version.split(".")[:2]) if version else None
        if major_minor and major_minor >= (3, 8):
            tools.append(
                ToolStatus(
                    name="Python 3.8+",
                    installed=True,
                    version=version,
                )
            )
        else:
            tools.append(
                ToolStatus(
                    name="Python 3.8+",
                    installed=False,
                    version=version,
                    detail="需要 Python 3.8 以上版本",
                )
            )
    else:
        tools.append(
            ToolStatus(
                name="Python 3.8+",
                installed=False,
                version=None,
                detail=stderr or "無法執行 python3 --version",
            )
        )

    return tools


def check_dependencies() -> DependencyReport:
    """執行完整依賴檢查並回傳報告。"""

    python_packages = _check_python_packages()
    system_tools = _check_system_tools()
    return DependencyReport(
        python_packages=python_packages,
        system_tools=system_tools,
    )


__all__ = [
    "DependencyReport",
    "PackageStatus",
    "ToolStatus",
    "check_dependencies",
]
