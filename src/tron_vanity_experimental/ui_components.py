"""
Rich UI 組件：提供 CLI 所需的表格與面板組裝函式。
"""

from __future__ import annotations

from typing import Sequence

try:
    from rich import box
    from rich.align import Align
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ModuleNotFoundError:  # pragma: no cover - 允許先載入模組再進行自動安裝
    box = None  # type: ignore
    Align = None  # type: ignore
    Panel = None  # type: ignore
    Table = None  # type: ignore
    Text = None  # type: ignore


def _ensure_rich() -> None:
    """確保 rich 元件可用，必要時動態載入。"""

    global box, Align, Panel, Table, Text
    if box is None or Align is None or Panel is None or Table is None or Text is None:
        try:
            from rich import box as _box
            from rich.align import Align as _Align
            from rich.panel import Panel as _Panel
            from rich.table import Table as _Table
            from rich.text import Text as _Text
        except ModuleNotFoundError as exc:
            raise RuntimeError("rich 尚未載入，請先安裝或透過 CLI 自動補齊後再使用 UI 組件") from exc
        box = _box
        Align = _Align
        Panel = _Panel
        Table = _Table
        Text = _Text

from .dependency_checker import DependencyReport
from .hardware_config import HardwareAdaptiveConfig
from .performance_estimator import DifficultyEstimate, SpeedEstimate
from .system_info import SystemInfo


_ASCII_LOGO = r"""
 _______ _______  ______  _   _      _   _      _______          _ _         
|__   __|__   __||  ____|| \ | |    | \ | |    |__   __|        | | |        
   | |     | |   | |__   |  \| | ___|  \| | ___   | | ___   ___ | | | ___ _ __ 
   | |     | |   |  __|  | . ` |/ _ \ . ` |/ _ \  | |/ _ \ / _ \| | |/ _ \ '__|
   | |     | |   | |____ | |\  |  __/ |\  |  __/  | | (_) | (_) | | |  __/ |   
   |_|     |_|   |______||_| \_|\___|_| \_|\___|  |_|\___/ \___/|_|_|\___|_|   
"""


def ascii_logo(version: str) -> Text:
    """回傳包含 ASCII Logo 與版本號的 Rich 文字物件。"""

    _ensure_rich()
    text = Text()
    text.append(_ASCII_LOGO.strip("\n"), style="bold cyan")
    text.append(f"\n\n  版本：v{version}", style="bold magenta")
    return text


def build_system_summary(
    system_info: SystemInfo,
    config: HardwareAdaptiveConfig,
    speed: SpeedEstimate,
) -> Panel:
    """建構硬體與性能摘要面板。"""

    _ensure_rich()
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", style="cyan", width=12)
    table.add_column(style="white")

    table.add_row("作業系統", f"{system_info.os_name} {system_info.os_version}")
    cpu_model = system_info.cpu.model or "未知型號"
    cpu_core_text = f"{system_info.cpu.physical_cores or '-'} / {system_info.cpu.logical_cores or '-'}"
    table.add_row("CPU 型號", cpu_model)
    table.add_row("CPU 核心", f"{cpu_core_text} (實體/邏輯)")

    if system_info.gpus:
        gpu_entries = [
            f"{gpu.name} ({gpu.memory_total_gb or 0:.1f} GiB)"
            for gpu in system_info.gpus
        ]
        table.add_row("GPU 列表", "\n".join(gpu_entries))
    else:
        table.add_row("GPU 列表", "未偵測到 NVIDIA GPU")

    mem_total = system_info.memory.total_gb or 0
    mem_avail = system_info.memory.available_gb
    mem_text = f"{mem_total:.1f} GiB"
    if mem_avail is not None:
        mem_text += f"（可用 {mem_avail:.1f} GiB）"
    table.add_row("記憶體", mem_text)

    table.add_row("Python 版本", system_info.python_version)

    table.add_row("運行模式", f"{config.backend} / {config.profile}")
    table.add_row(
        "批次大小",
        f"預設 {config.default_batches}，上限 {config.max_batch_size}",
    )
    table.add_row(
        "CPU 建議進程",
        f"{config.recommended_processes}（max pending x{config.max_pending_multiplier}）",
    )
    if config.backend == "GPU":
        table.add_row(
            "CUDA Streams",
            f"預設 {config.default_streams} 條（可依需求調整）",
        )
    table.add_row(
        "估計吞吐",
        f"{int(speed.addr_per_sec):,} addr/s · {speed.label}",
    )
    if speed.notes:
        table.add_row("備註", "\n".join(speed.notes))

    return Panel(
        Align.left(table),
        title="硬體摘要",
        border_style="bright_cyan",
        box=box.ROUNDED,
    )


def build_dependency_table(report: DependencyReport) -> Panel:
    """建構依賴檢測結果面板。"""

    _ensure_rich()
    table = Table(
        title="Python 套件",
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE,
    )
    table.add_column("項目", style="white", no_wrap=True)
    table.add_column("類別", style="cyan", no_wrap=True)
    table.add_column("狀態", style="green")
    table.add_column("版本/封裝", style="magenta")
    table.add_column("備註", style="yellow")

    for pkg in report.python_packages:
        status = "[green]已安裝[/green]" if pkg.installed else "[red]缺失[/red]"
        version = pkg.version or "-"
        if pkg.package_name and pkg.package_name != pkg.display_name:
            version = f"{version}（{pkg.package_name}）"
        table.add_row(
            pkg.display_name,
            "必要" if pkg.required else "選用",
            status,
            version,
            pkg.detail or "-",
        )

    tool_table = Table(
        title="系統工具",
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE,
    )
    tool_table.add_column("項目", style="white", no_wrap=True)
    tool_table.add_column("類別", style="cyan", no_wrap=True)
    tool_table.add_column("狀態", style="green")
    tool_table.add_column("版本", style="magenta")
    tool_table.add_column("備註", style="yellow")

    for tool in report.system_tools:
        status = "[green]可用[/green]" if tool.installed else "[red]缺失[/red]"
        tool_table.add_row(
            tool.name,
            "必要",
            status,
            tool.version or "-",
            tool.detail or "-",
        )

    grid = Table.grid()
    grid.add_row(table)
    grid.add_row(tool_table)
    return Panel(Align.left(grid), title="依賴檢測", border_style="bright_green", box=box.ROUNDED)


def build_manual_actions_panel(actions: Sequence[str]) -> Panel:
    """輸出需人工處理的提示面板。"""

    _ensure_rich()
    content = "\n".join(f"- {item}" for item in actions)
    return Panel(
        Align.left(Text(content or "無需額外操作", style="white")),
        title="人工操作建議",
        border_style="yellow",
        box=box.ROUNDED,
    )


def build_difficulty_panel(target: str, difficulty: DifficultyEstimate) -> Panel:
    """顯示靚號搜尋難度與預估時間。"""

    _ensure_rich()
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", style="cyan", width=12)
    table.add_column(style="white")
    table.add_row("目標尾碼", target or "(未設定)")
    table.add_row("字元數", str(difficulty.suffix_length))
    table.add_row("預估嘗試", f"{difficulty.attempts:,} 次")
    table.add_row("預估時間", difficulty.human_time)
    table.add_row("難度等級", difficulty.difficulty)
    table.add_row("可行性", "✅ 可行" if difficulty.feasible else "⚠️ 需長時間")
    return Panel(
        Align.left(table),
        title="靚號難度評估",
        border_style="bright_magenta",
        box=box.ROUNDED,
    )


__all__ = [
    "ascii_logo",
    "build_system_summary",
    "build_dependency_table",
    "build_manual_actions_panel",
    "build_difficulty_panel",
]
