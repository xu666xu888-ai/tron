"""
產品級 CLI 主入口：整合系統檢測、自動安裝與性能預估。
後續將擴充實際靚號搜尋與監控流程。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List
from collections import deque
import time
from rich.table import Table

if __package__ in (None, "", "__main__"):
    # 允許直接以 `python src/tron_vanity/cli.py` 執行
    repo_src = Path(__file__).resolve().parents[1]
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    __package__ = "tron_vanity"

from . import __version__
DEFAULT_CONFIG_LOCATIONS = (
    Path.cwd() / "tron_vanity_cli.json",
    Path.home() / ".tron_vanity_cli.json",
)
HISTORY_PATH = Path.home() / ".tron_vanity_history.jsonl"

CONFIG_KEYS = (
    "suffix",
    "timeout",
    "max_attempts",
    "cpu_only",
    "output",
    "auto_install",
    "auto_confirm",
    "monitor",
)


from .auto_installer import ensure_python_dependencies, plan_manual_actions
from .dependency_checker import DependencyReport, check_dependencies
from .hardware_config import HardwareAdaptiveConfig, detect_hardware_config
from .performance_estimator import (
    DifficultyEstimate,
    SpeedEstimate,
    estimate_speed,
    estimate_suffix_difficulty,
)
from .system_info import SystemInfo, collect_system_info
from .monitor import VanitySearchMonitor
from .search_engine import SearchResult, VanitySearchEngine
from .ui_components import (
    ascii_logo,
    build_dependency_table,
    build_difficulty_panel,
    build_manual_actions_panel,
    build_system_summary,
)

Console = None
Panel = None


def _import_rich() -> bool:
    """嘗試載入 rich，若失敗則回傳 False。"""

    global Console, Panel
    try:
        from rich.console import Console  # type: ignore
        from rich.panel import Panel  # type: ignore

        return True
    except ModuleNotFoundError:
        return False


def _prepare_console(allow_auto_install: bool = True) -> "Console":
    """確保 rich 可用並回傳 Console 實例。"""

    if not _import_rich():
        if not allow_auto_install:
            raise SystemExit("未安裝 rich，請先執行 `pip install rich` 或移除 --no-auto-install 選項。")
        print("未偵測到 rich，正在安裝 CLI 介面依賴...")
        report = ensure_python_dependencies(include_optional=True)
        if not _import_rich():
            raise SystemExit("rich 安裝失敗，請手動執行 `pip install rich` 後重試。")
        if report.manual_actions:
            for action in report.manual_actions:
                print(f"[提示] {action}")
    return Console()


def _ensure_config_structure(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config) if isinstance(config, dict) else {}
    presets = cfg.get("presets")
    if not isinstance(presets, dict):
        presets = {}
    cfg["presets"] = presets
    cfg.setdefault("default_preset", None)
    return cfg


def _collect_settings_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "suffix": args.suffix,
        "timeout": args.timeout,
        "max_attempts": args.max_attempts,
        "cpu_only": bool(args.cpu_only),
        "output": args.output,
        "auto_install": not args.no_auto_install,
        "auto_confirm": bool(args.yes),
        "monitor": not args.no_monitor,
    }


def _render_dependency_section(console: "Console", report: DependencyReport) -> None:
    """輸出依賴檢測結果。"""

    console.print(build_dependency_table(report))


def _render_system_summary(
    console: "Console",
    system_info: SystemInfo,
    *,
    force_cpu: bool = False,
) -> Tuple[HardwareAdaptiveConfig, SpeedEstimate]:
    """輸出系統與硬體摘要。"""

    hw_config = detect_hardware_config(force_cpu=force_cpu)
    speed = estimate_speed(hw_config)
    console.print(build_system_summary(system_info, hw_config, speed))
    return hw_config, speed


def _render_difficulty(console: "Console", suffix: str, estimate: DifficultyEstimate) -> None:
    """顯示難度面板。"""

    console.print(build_difficulty_panel(suffix, estimate))


def _save_result(
    result: SearchResult,
    suffix: str,
    output_path: Optional[str],
    system_info: SystemInfo,
    config: HardwareAdaptiveConfig,
) -> Optional[Path]:
    """將命中結果寫入指定檔案（JSON Lines）。"""

    if not output_path or not result.hits:
        return None
    path = Path(output_path).expanduser()
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
        path = path / "tron_vanity_hits.jsonl"
    record = {
        "suffix": suffix,
        "backend": result.backend,
        "attempts": result.attempts,
        "elapsed_seconds": result.elapsed,
        "address_hex": result.hits[0].address_hex,
        "address_base58": result.hits[0].address_base58,
        "privkey_hex": result.hits[0].privkey_hex,
        "system": {
            "os": f"{system_info.os_name} {system_info.os_version}",
            "cpu": system_info.cpu.model,
            "gpu": [
                {"name": gpu.name, "memory_gb": gpu.memory_total_gb, "cuda": gpu.cuda_version}
                for gpu in system_info.gpus
            ],
        },
        "config": {
            "backend": config.backend,
            "profile": config.profile,
            "default_batches": config.default_batches,
            "max_batch_size": config.max_batch_size,
        },
    }
    with path.open("a", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False)
        fh.write("\n")
    return path


def _merge_settings_into_args(
    args: argparse.Namespace,
    settings: Dict[str, Any],
    user_flags: Dict[str, bool],
    meta: Dict[str, bool],
) -> None:
    if not settings:
        return

    if args.suffix is None and settings.get("suffix"):
        args.suffix = str(settings["suffix"])
        meta["suffix_from_config"] = True

    if args.timeout is None and settings.get("timeout") is not None:
        args.timeout = _as_float(settings.get("timeout"))

    if args.max_attempts is None and settings.get("max_attempts") is not None:
        args.max_attempts = _as_int(settings.get("max_attempts"))

    if not args.cpu_only and settings.get("cpu_only"):
        args.cpu_only = True

    if args.output is None and settings.get("output"):
        args.output = str(settings["output"])

    if not user_flags.get("no_auto_install_cli"):
        auto_install = settings.get("auto_install")
        if auto_install is not None:
            args.no_auto_install = not bool(auto_install)
            meta["auto_install_set"] = True

    if not user_flags.get("yes_cli"):
        auto_confirm = settings.get("auto_confirm")
        if auto_confirm is not None:
            args.yes = bool(auto_confirm)
            meta["auto_confirm_set"] = True
            meta["yes_from_config"] = bool(auto_confirm)

    monitor_setting = settings.get("monitor")
    if monitor_setting is not None and monitor_setting is False:
        args.no_monitor = True


def _record_history(
    suffix: str,
    result: SearchResult,
    preset: Optional[str],
    config_path: Optional[Path],
    output_path: Optional[Path],
) -> None:
    history_entry = {
        "suffix": suffix,
        "status": "found" if result.found else result.reason,
        "address": result.hits[0].address_base58 if result.hits else None,
        "privkey": result.hits[0].privkey_hex if result.hits else None,
        "attempts": result.attempts,
        "elapsed": result.elapsed,
        "backend": result.backend,
        "preset": preset,
        "config": str(config_path) if config_path else None,
        "output_file": str(output_path) if output_path else None,
        "timestamp": time.time(),
    }
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as fh:
        json.dump(history_entry, fh, ensure_ascii=False)
        fh.write("\n")


def _display_history(console: "Console", limit: Optional[int]) -> None:
    if not HISTORY_PATH.exists():
        console.print("[yellow]尚無搜尋紀錄。[/yellow]")
        return
    entries = deque(maxlen=limit or 10)
    with HISTORY_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not entries:
        console.print("[yellow]尚無搜尋紀錄。[/yellow]")
        return
    table = Table(title="搜尋紀錄", header_style="bold cyan")
    table.add_column("時間")
    table.add_column("尾碼")
    table.add_column("狀態")
    table.add_column("地址")
    table.add_column("嘗試數")
    table.add_column("耗時(s)")
    table.add_column("Preset")
    for entry in reversed(entries):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.get("timestamp", 0)))
        suffix = entry.get("suffix") or "-"
        status = entry.get("status") or "-"
        address = entry.get("address") or "-"
        attempts = f"{entry.get('attempts', 0):,}"
        elapsed = f"{entry.get('elapsed', 0):.2f}"
        preset = entry.get("preset") or "-"
        table.add_row(ts, suffix, status, address, attempts, elapsed, preset)
    console.print(table)


def _clear_history(console: "Console") -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
        console.print("[green]已清除搜尋紀錄。[/green]")
    else:
        console.print("[yellow]目前沒有紀錄檔可清除。[/yellow]")


def _list_presets(console: "Console", config: Dict[str, Any]) -> None:
    config = _ensure_config_structure(config)
    presets = config.get("presets", {})
    if not presets:
        console.print("[yellow]尚未建立任何預設設定。[/yellow]")
        return
    table = Table(title="預設設定", header_style="bold cyan")
    table.add_column("名稱", style="white")
    table.add_column("尾碼", style="cyan")
    table.add_column("超時(s)", style="magenta")
    table.add_column("最大嘗試", style="magenta")
    table.add_column("模式", style="green")
    table.add_column("輸出", style="yellow")
    default_preset = config.get("default_preset")
    for name, settings in presets.items():
        suffix = settings.get("suffix", "-")
        timeout = settings.get("timeout")
        max_attempts = settings.get("max_attempts")
        cpu_only = "CPU" if settings.get("cpu_only") else "GPU"
        output = settings.get("output") or "-"
        label = f"{name} (預設)" if name == default_preset else name
        table.add_row(
            label,
            str(suffix or "-"),
            "-" if timeout is None else str(timeout),
            "-" if max_attempts is None else f"{max_attempts:,}",
            cpu_only,
            output,
        )
    console.print(table)


def _prompt_suffix(
    console: "Console",
    default: Optional[str],
    *,
    skip_prompt: bool,
) -> str:
    """互動詢問尾碼；若 skip_prompt 為 True 則直接回傳預設值。"""

    default = (default or "").strip()
    interactive = sys.stdin.isatty()
    if skip_prompt or not interactive:
        if not default and not interactive:
            raise SystemExit("偵測到非互動環境，請使用 --suffix 或預設設定指定尾碼。")
        if not default:
            return ""
        if not interactive and not skip_prompt:
            console.print(f"[yellow]非互動環境偵測，使用預設尾碼：{default}[/yellow]")
        return default

    while True:
        if default:
            prompt = (
                "[bold cyan]請輸入欲搜尋的末幾碼 (例如 88888)，直接 Enter 使用預設 "
                f"[white]{default}[/white]：[/] "
            )
        else:
            prompt = "[bold cyan]請輸入欲搜尋的末幾碼 (例如 88888)：[/] "
        value = console.input(prompt).strip()
        if value:
            return value
        if default:
            console.print(f"[green]已套用預設尾碼：{default}[/green]")
            return default
        console.print("[red]尾碼不可為空白，請重新輸入。[/red]")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _load_cli_config(path: Optional[str]) -> Tuple[Dict[str, Any], Optional[Path]]:
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    else:
        candidates.extend(DEFAULT_CONFIG_LOCATIONS)

    for candidate in candidates:
        if candidate.is_file():
            try:
                with candidate.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    return _ensure_config_structure(data), candidate
                raise ValueError("配置內容需為 JSON 物件")
            except Exception as exc:
                raise SystemExit(f"讀取配置檔案失敗：{candidate} ({exc})") from exc
    return _ensure_config_structure({}), None


def _apply_config(
    args: argparse.Namespace,
    config: Dict[str, Any],
    user_flags: Dict[str, bool],
) -> Dict[str, bool]:
    config = _ensure_config_structure(config)
    meta = {
        "suffix_from_config": False,
        "auto_confirm_set": False,
        "auto_install_set": False,
        "yes_from_config": False,
        "config_loaded": bool(config),
        "active_preset": None,
        "preset_from_cli": user_flags.get("preset_cli", False),
    }
    base_settings = {key: config.get(key) for key in CONFIG_KEYS}
    _merge_settings_into_args(args, base_settings, user_flags, meta)

    presets = config.get("presets", {})
    preset_name = args.preset or config.get("default_preset")
    if preset_name:
        preset_settings = presets.get(preset_name)
        if preset_settings is None:
            if user_flags.get("preset_cli"):
                raise SystemExit(f"找不到預設檔：{preset_name}")
        else:
            _merge_settings_into_args(args, preset_settings, user_flags, meta)
            meta["active_preset"] = preset_name
            args.preset = preset_name

    return meta


def _write_config(config: Dict[str, Any], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
    return target


def _save_cli_config(
    config: Dict[str, Any],
    path: Optional[str],
    args: argparse.Namespace,
    *,
    preset: Optional[str] = None,
    set_default: bool = False,
) -> Path:
    cfg = _ensure_config_structure(config)
    target = Path(path).expanduser() if path else DEFAULT_CONFIG_LOCATIONS[-1]
    settings = _collect_settings_from_args(args)
    if preset:
        cfg["presets"][preset] = settings
        if set_default or cfg.get("default_preset") is None:
            cfg["default_preset"] = preset
    else:
        for key, value in settings.items():
            cfg[key] = value
        if set_default and args.preset:
            cfg["default_preset"] = args.preset
    return _write_config(cfg, target)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRON 靚號地址生成 CLI")
    parser.add_argument("--suffix", type=str, help="直接指定欲搜尋的 Base58 尾碼（例如 88888）")
    parser.add_argument("--timeout", type=float, help="設定搜尋逾時秒數")
    parser.add_argument("--max-attempts", type=int, help="限制最多嘗試次數")
    parser.add_argument("--cpu-only", action="store_true", help="強制使用 CPU 模式搜尋")
    parser.add_argument("--output", type=str, help="將命中結果追加寫入檔案（JSON Lines）；可為資料夾")
    parser.add_argument("--no-auto-install", action="store_true", help="避免自動安裝缺失的 Python 套件")
    parser.add_argument("--yes", action="store_true", help="跳過互動確認並立即開始搜尋")
    parser.add_argument("--no-confirm", dest="yes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-monitor", action="store_true", help="停用 Rich 監控介面（純文字輸出）")
    parser.add_argument("--preset", type=str, help="套用指定的預設設定名稱")
    parser.add_argument("--save-preset", type=str, help="將目前參數儲存為指定名稱的預設設定")
    parser.add_argument("--list-presets", action="store_true", help="列出所有可用預設設定")
    parser.add_argument("--set-default-preset", type=str, help="設定預設預設設定名稱")
    parser.add_argument("--config", type=str, help="指定配置檔路徑（JSON）")
    parser.add_argument(
        "--show-history",
        nargs="?",
        type=int,
        const=10,
        help="顯示最近的搜尋紀錄，可指定要顯示的筆數（預設 10）",
    )
    parser.add_argument("--clear-history", action="store_true", help="清除搜尋紀錄檔")
    parser.add_argument(
        "--save-config",
        nargs="?",
        const="",
        help="將目前參數存成配置檔；若未提供路徑，預設寫入 ~/.tron_vanity_cli.json",
    )
    return parser.parse_args([] if argv is None else list(argv))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 主流程。"""

    args = _parse_args(argv)
    user_flags = {
        "yes_cli": bool(args.yes),
        "no_auto_install_cli": bool(args.no_auto_install),
        "preset_cli": args.preset is not None,
    }
    config, config_path = _load_cli_config(args.config)
    meta = _apply_config(args, config, user_flags)
    auto_install = not args.no_auto_install
    console = _prepare_console(allow_auto_install=auto_install)
    config = _ensure_config_structure(config)

    pending_default_preset: Optional[str] = None

    if args.list_presets:
        _list_presets(console, config)
        return 0

    if args.set_default_preset:
        preset_name = args.set_default_preset
        if preset_name not in config.get("presets", {}):
            if args.save_preset == preset_name:
                pending_default_preset = preset_name
            else:
                console.print(f"[red]找不到預設設定：{preset_name}[/red]")
                return 1
        else:
            config["default_preset"] = preset_name
            target_path = Path(args.config).expanduser() if args.config else (config_path or DEFAULT_CONFIG_LOCATIONS[-1])
            _write_config(config, target_path)
            console.print(f"[green]已將預設預設設定設為：{preset_name}[/green]")
            config_path = target_path

    if args.clear_history:
        _clear_history(console)
        return 0

    if args.show_history is not None:
        _display_history(console, args.show_history)
        return 0

    console.clear()
    if config_path:
        console.print(f"[cyan]載入 CLI 設定：{config_path}[/cyan]")
    console.print(ascii_logo(__version__), justify="center")
    console.rule("系統檢測")

    with console.status("[green]正在收集系統資訊...[/]"):
        system_info = collect_system_info()

    with console.status("[green]正在檢查依賴...[/]"):
        dep_report = check_dependencies()

    manual_actions: List[str] = plan_manual_actions(dep_report)
    if auto_install and not dep_report.all_satisfied():
        console.print("[yellow]偵測到缺失依賴，啟動自動安裝...[/yellow]")
        install_report = ensure_python_dependencies(include_optional=False)
        dep_report = install_report.dependency_report
        manual_actions = install_report.manual_actions or plan_manual_actions(dep_report)

    _render_dependency_section(console, dep_report)
    if manual_actions:
        console.print(build_manual_actions_panel(manual_actions))
    required_missing = [pkg.display_name for pkg in dep_report.python_packages if pkg.required and not pkg.installed]
    optional_missing = [pkg.display_name for pkg in dep_report.python_packages if (not pkg.required) and (not pkg.installed)]
    required_missing += [tool.name for tool in dep_report.system_tools if not tool.installed]
    if required_missing:
        console.print(
            f"[red]仍缺少必要依賴：{', '.join(required_missing)}，請依上方指引補齊後再進行高強度測試。[/red]"
        )
    elif optional_missing:
        console.print(
            f"[yellow]以下為選用功能，可視需求安裝：{', '.join(optional_missing)}。[/yellow]"
        )
    hw_config, speed = _render_system_summary(console, system_info, force_cpu=args.cpu_only)
    if meta.get("active_preset"):
        console.print(f"[cyan]套用預設設定：{meta['active_preset']}[/cyan]")

    console.rule("靚號設定")
    skip_suffix_prompt = args.yes and not meta.get("yes_from_config", False)
    suffix = _prompt_suffix(console, args.suffix, skip_prompt=skip_suffix_prompt)
    suffix = suffix.strip()
    if not suffix:
        console.print("[red]未提供尾碼，無法開始搜尋。[/red]")
        return 1
    args.suffix = suffix

    difficulty = estimate_suffix_difficulty(len(suffix), hw_config)
    _render_difficulty(console, suffix, difficulty)

    if args.save_config is not None:
        save_path = _save_cli_config(config, args.save_config or None, args)
        console.print(f"[green]已將設定寫入：{save_path}[/green]")
        config_path = save_path

    console.rule("開始搜尋")
    skip_confirm = args.yes and not meta.get("yes_from_config", False)
    if not skip_confirm:
        if not sys.stdin.isatty():
            console.print("[yellow]非互動環境偵測，已自動開始搜尋。[/yellow]")
        else:
            confirmation = console.input("[bold green]按 Enter 開始搜尋，輸入 N 取消：[/] ").strip().lower()
            if confirmation in {"n", "no"}:
                console.print("[bold yellow]已取消搜尋。[/bold yellow]")
                return 0

    if args.cpu_only and hw_config.backend != "CPU":
        hw_config = detect_hardware_config(force_cpu=True)
    engine = VanitySearchEngine(hw_config)

    monitor: Optional[VanitySearchMonitor] = None
    if not args.no_monitor:
        monitor = VanitySearchMonitor(console, suffix)

    def _progress(checked: int, hits: int, last_batch: int, elapsed: float) -> None:
        if monitor is not None:
            monitor.update(checked=checked, hits=hits, last_batch=last_batch, elapsed=elapsed)
        else:
            message = (
                f"[progress]已檢查 {checked:,} 筆，最近速率 {last_batch / max(elapsed, 1e-6):,.0f} addr/s[/progress]"
            )
            console.print(message, end="\r")

    console.print("[cyan]初始設定完畢，開始搜尋靚號…[/cyan]")
    result: Optional[SearchResult] = None
    output_saved_path: Optional[Path] = None
    try:
        if monitor is not None:
            monitor.start()
        result = engine.search_suffix(
            suffix,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            progress_callback=_progress,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]使用者中止搜尋。[/yellow]")
        return 130
    finally:
        if monitor is not None:
            monitor.stop()
        else:
            console.print()

    console.rule("搜尋結果")
    if not result:
        console.print("[red]搜尋流程結束，但未取得結果資訊。[/red]")
        return 1

    if result.found and result.hits:
        hit = result.hits[0]
        summary = (
            f"[bold green]命中靚號！[/bold green]\n"
            f"後端：{result.backend}\n"
            f"耗時：{result.elapsed:.2f} 秒\n"
            f"嘗試次數：約 {result.attempts:,} 次\n"
            f"Base58 地址：{hit.address_base58}\n"
            f"HEX 地址：{hit.address_hex}\n"
            f"私鑰 (HEX)：{hit.privkey_hex}"
        )
        console.print(Panel(summary, border_style="green", title="成功"))
        console.print("[bold cyan]請妥善保存上述私鑰與地址資訊。[/bold cyan]")
        output_saved_path = _save_result(result, suffix, args.output, system_info, hw_config)
        if output_saved_path:
            console.print(f"[green]已將結果寫入：{output_saved_path}[/green]")
        recap_preset = meta.get("active_preset") or args.preset or args.save_preset
        if args.save_preset:
            preset_path = _save_cli_config(
                config,
                args.config,
                args,
                preset=args.save_preset,
                set_default=(pending_default_preset == args.save_preset),
            )
            console.print(f"[green]已更新預設設定：{args.save_preset}（檔案：{preset_path}）[/green]")
            if pending_default_preset == args.save_preset:
                console.print(f"[cyan]已同步將 {args.save_preset} 設為預設設定。[/cyan]")
            config_path = preset_path
            recap_preset = args.save_preset
        _record_history(suffix, result, recap_preset, config_path, output_saved_path)
        return 0

    reason_map = {
        "timeout": "已達設定的逾時限制",
        "max_attempts": "已達最大嘗試次數限制",
        "found": "搜尋完成",
    }
    reason_text = reason_map.get(result.reason, result.reason)
    console.print(
        Panel(
            f"[yellow]未命中靚號。[/yellow]\n"
            f"原因：{reason_text}\n"
            f"耗時：{result.elapsed:.2f} 秒\n"
            f"嘗試次數：約 {result.attempts:,} 次",
            border_style="yellow",
            title="未命中",
        )
    )
    console.print("[bold white]可調整尾碼長度或於設定中加入逾時/最大嘗試限制後再試。[/bold white]")
    recap_preset = meta.get("active_preset") or args.preset or args.save_preset
    if args.save_preset:
        preset_path = _save_cli_config(
            config,
            args.config,
            args,
            preset=args.save_preset,
            set_default=(pending_default_preset == args.save_preset),
        )
        console.print(f"[green]已更新預設設定：{args.save_preset}（檔案：{preset_path}）[/green]")
        if pending_default_preset == args.save_preset:
            console.print(f"[cyan]已同步將 {args.save_preset} 設為預設設定。[/cyan]")
        config_path = preset_path
        recap_preset = args.save_preset
    _record_history(suffix, result, recap_preset, config_path, None)
    return 2


def _signal_handler(sig, frame) -> None:
    """優雅處理 Ctrl+C。"""

    print("\n[CLI] 已接收中斷訊號，結束程序。")
    sys.exit(0)


def run() -> None:
    """提供 `python -m tron_vanity.cli` 的入口。"""

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    exit_code = main(sys.argv[1:])
    raise SystemExit(exit_code)


if __name__ == "__main__":
    run()
