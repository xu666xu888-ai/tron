"""
自動安裝模組：針對缺失的 Python 依賴執行 pip 安裝，並提供系統級工具的人工安裝指引。
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .dependency_checker import (
    DependencyReport,
    PackageStatus,
    ToolStatus,
    check_dependencies,
)
from .system_info import SystemInfo, collect_system_info

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    _RICH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - 若尚未安裝 rich 也能繼續執行
    Console = None  # type: ignore
    Progress = None  # type: ignore
    _RICH_AVAILABLE = False


_PACKAGE_NAME_MAPPING = {
    "tronpy": "tronpy",
    "ecdsa": "ecdsa",
    "base58": "base58",
    "rich": "rich",
    "psutil": "psutil",
    "GPUtil": "gputil",
}


@dataclass
class InstallResult:
    """紀錄單一套件的安裝結果。"""

    package: str
    success: bool
    message: Optional[str] = None


@dataclass
class AutoInstallReport:
    """自動安裝後的彙整資訊。"""

    dependency_report: DependencyReport
    install_results: List[InstallResult] = field(default_factory=list)
    manual_actions: List[str] = field(default_factory=list)

    def succeeded(self) -> bool:
        """判斷是否所有必要依賴皆已就緒。"""

        installs_ok = all(result.success for result in self.install_results)
        deps_ok = self.dependency_report.all_satisfied()
        return installs_ok and deps_ok


def _select_cupy_package(system_info: Optional[SystemInfo]) -> Tuple[str, str]:
    """
    根據 CUDA 版本推斷應安裝的 CuPy 發行版。

    回傳 (pip 名稱, 說明字串)。
    """

    if system_info:
        for gpu in system_info.gpus:
            if gpu.cuda_version:
                major = gpu.cuda_version.split(".")[0]
                if major.isdigit():
                    major_num = int(major)
                    if major_num >= 13:
                        return "cupy-cuda12x", (
                            f"偵測到 CUDA {gpu.cuda_version}，目前僅支援 CUDA 12 系列，將安裝 cupy-cuda12x"
                        )
                    if major_num >= 12:
                        return "cupy-cuda12x", f"偵測到 CUDA {gpu.cuda_version}"
                    if major_num == 11:
                        return "cupy-cuda11x", f"偵測到 CUDA {gpu.cuda_version}"
    # 無法偵測時預設安裝 CUDA 12 版本，並提示使用者確認
    return "cupy-cuda12x", "未偵測到 GPU/CUDA 版本，預設使用 CUDA 12 版"


def _pip_install(package: str) -> InstallResult:
    """使用 pip 安裝指定套件。"""

    if (
        sys.platform.startswith("win")
        and sys.version_info >= (3, 13)
        and (package.lower().startswith("cupy") or package.lower().startswith("numpy") or package.lower() == "cupy")
    ):
        message = (
            "Windows + Python 3.13 尚無預編譯 wheel，請改用 Python 3.10~3.12 並安裝對應的 cupy/numpy wheel。"
        )
        return InstallResult(package=package, success=False, message=message)

    command = [sys.executable, "-m", "pip", "install", package]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    success = result.returncode == 0
    message = result.stdout if success else result.stderr
    return InstallResult(package=package, success=success, message=message)


def _add_manual_actions(report: AutoInstallReport) -> None:
    """針對缺失的系統工具附上建議操作。"""

    existing = set(report.manual_actions)

    for tool in report.dependency_report.system_tools:
        if tool.installed:
            continue
        if "CUDA Toolkit" in tool.name:
            msg = "未偵測到 CUDA Toolkit，請依官方指引安裝： https://developer.nvidia.com/cuda-downloads"
            if msg not in existing:
                report.manual_actions.append(msg)
                existing.add(msg)
        elif "NVIDIA Driver" in tool.name:
            msg = "未偵測到 NVIDIA Driver，請安裝適用 GPU 的驅動程式與 nvidia-smi"
            if msg not in existing:
                report.manual_actions.append(msg)
                existing.add(msg)
        elif "Python 3.8+" in tool.name:
            msg = "Python 版本低於 3.8，請升級至 3.8 以上版本後再執行"
            if msg not in existing:
                report.manual_actions.append(msg)
                existing.add(msg)

    for pkg in report.dependency_report.python_packages:
        if pkg.installed or pkg.required:
            continue
        msg = f"可選套件 {pkg.display_name} 尚未安裝，可視需求執行 `pip install {pkg.package_name or pkg.display_name}`"
        if msg not in existing:
            report.manual_actions.append(msg)
            existing.add(msg)

    for res in report.install_results:
        if res.success:
            continue
        detail = f"：{res.message}" if res.message else ""
        msg = f"安裝 {res.package} 失敗{detail}。請手動安裝或調整環境後再執行 `pip install {res.package}`"
        if msg not in existing:
            report.manual_actions.append(msg)
            existing.add(msg)


def ensure_python_dependencies(
    include_optional: bool = True,
) -> AutoInstallReport:
    """
    自動安裝缺失的 Python 依賴。

    include_optional: 是否一併安裝選用套件（例如 GPUtil）。
    """

    system_info = collect_system_info()
    dependency_report = check_dependencies()
    report = AutoInstallReport(dependency_report=dependency_report)

    missing_packages: List[PackageStatus] = []
    for pkg in dependency_report.python_packages:
        if not pkg.installed and (pkg.required or include_optional):
            missing_packages.append(pkg)

    if not missing_packages:
        _add_manual_actions(report)
        return report

    console: Optional[Console] = Console() if _RICH_AVAILABLE else None
    progress: Optional[Progress] = None
    task_id = None
    if _RICH_AVAILABLE:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        )
        progress.start()
        task_id = progress.add_task("安裝依賴中...", total=len(missing_packages))
    else:
        print("正在安裝缺失的 Python 套件...")  # pragma: no cover - 文字模式提示

    for pkg_status in missing_packages:
        pip_name = _PACKAGE_NAME_MAPPING.get(pkg_status.display_name, pkg_status.display_name)
        note = None

        if pkg_status.display_name.startswith("CuPy"):
            pip_name, note = _select_cupy_package(system_info)

        if console and note:
            console.log(f"[cyan]{note}[/cyan]")
        elif note:
            print(note)  # pragma: no cover

        result = _pip_install(pip_name)
        report.install_results.append(result)

        if progress and task_id is not None:
            progress.advance(task_id)
            progress.refresh()

        if console:
            if result.success:
                console.log(f"[green]已安裝 {pip_name}[/green]")
            else:
                console.log(f"[red]安裝 {pip_name} 失敗[/red]")
                if result.message:
                    snippet = result.message.strip().splitlines()
                    snippet = snippet[:8]
                    console.log("\n".join(snippet))
        elif not result.success:
            print(f"安裝 {pip_name} 失敗")  # pragma: no cover
            if result.message:
                print(result.message)  # pragma: no cover

    if progress:
        progress.stop()

    # 重新檢測依賴狀態，更新報告
    report.dependency_report = check_dependencies()
    _add_manual_actions(report)

    return report


def plan_manual_actions(
    dependency_report: DependencyReport,
    install_results: Optional[List[InstallResult]] = None,
) -> List[str]:
    """根據依賴檢測結果輸出建議操作（不執行安裝）。"""

    report = AutoInstallReport(
        dependency_report=dependency_report,
        install_results=list(install_results or []),
    )
    _add_manual_actions(report)
    return report.manual_actions


__all__ = [
    "AutoInstallReport",
    "InstallResult",
    "ensure_python_dependencies",
    "plan_manual_actions",
]
