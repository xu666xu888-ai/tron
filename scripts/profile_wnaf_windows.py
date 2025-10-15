# -*- coding: utf-8 -*-
"""
GPU wNAF Window Profiling
=========================

快速比較不同 secp256k1 kernel （Baseline、Window4、Window6、Window6-GLV、Window8）
的吞吐量與耗時，便於後續優化 / profiling 參考。

Usage
-----
    PYTHONPATH=src python scripts/profile_wnaf_windows.py \
        --batches 8192 16384 32768 \
        --repeats 5 \
        --windows baseline w4 w6 w6_glv

輸出為人類可讀表格與 JSON 結構，方便複製到報告或作圖。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import importlib

import cupy as cp

# 允許腳本在 repo 根目錄直接執行
if __package__ in (None, "", "__main__"):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    __package__ = "scripts"

WindowFunc = Callable[[cp.ndarray], cp.ndarray]


def _sync() -> None:
    """確保 GPU 工作完成，避免主機時間測不到 kernel 結束。"""
    cp.cuda.Stream.null.synchronize()


def _measure_once(func: WindowFunc, secrets: cp.ndarray) -> float:
    """量測單次执行耗時（秒）。"""
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    func(secrets)
    end.record()
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end)) / 1000.0


def _measure_window(label: str, func: WindowFunc, batch: int, repeats: int) -> Dict[str, float]:
    """針對指定 kernel 進行多次量測，回傳平均耗時與吞吐量。"""
    secrets = cp.random.randint(0, 256, size=(batch, 32), dtype=cp.uint8)
    # 預熱一次，避免第一輪載入常數等開銷影響
    func(secrets)
    _sync()
    samples: List[float] = []
    for _ in range(repeats):
        elapsed = _measure_once(func, secrets)
        samples.append(elapsed)
    avg = statistics.mean(samples)
    std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    throughput = batch / avg
    return {
        "label": label,
        "batch": batch,
        "repeats": repeats,
        "avg_sec": avg,
        "std_sec": std,
        "throughput": throughput,
        "samples": samples,
    }


def _select_windows(names: Iterable[str], mapping: Dict[str, Tuple[str, WindowFunc]]) -> List[Tuple[str, WindowFunc]]:
    result: List[Tuple[str, WindowFunc]] = []
    for name in names:
        key = name.lower()
        if key not in mapping:
            raise ValueError(f"未知的 window 名稱：{name}")
        result.append(mapping[key])
    return result


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile secp256k1 wNAF kernels")
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=[8192, 16384, 32768],
        help="要測試的批次大小（可多個）",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="每個批次重複次數",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["baseline", "w4", "w6", "w6_glv", "w8"],
        help="要測試的 kernel 名稱，從 baseline/w4/w6/w6_glv/w8 中選擇",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="選填，將結果輸出成 JSON 檔",
    )
    parser.add_argument(
        "--module",
        default="tron_vanity.gpu_secp256k1",
        help="指定要測試的 GPU 模組（預設 tron_vanity.gpu_secp256k1）",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    batches = [b for b in args.batches if b > 0]
    if not batches:
        raise ValueError("至少需指定一個正整數批次大小")
    repeats = max(1, args.repeats)
    module = importlib.import_module(args.module)

    gpu_secp256k1_batch = getattr(module, "gpu_secp256k1_batch")
    gpu_secp256k1_batch_window4 = getattr(module, "gpu_secp256k1_batch_window4")
    gpu_secp256k1_batch_window6 = getattr(module, "gpu_secp256k1_batch_window6")
    gpu_secp256k1_batch_window6_glv = getattr(module, "gpu_secp256k1_batch_window6_glv")
    gpu_secp256k1_batch_window8 = getattr(module, "gpu_secp256k1_batch_window8")
    warmup_window4_table = getattr(module, "warmup_window4_table")
    warmup_window6_table = getattr(module, "warmup_window6_table")
    warmup_window8_table = getattr(module, "warmup_window8_table")

    warmup_window4_table()
    warmup_window6_table()
    warmup_window8_table()

    mapping: Dict[str, Tuple[str, WindowFunc]] = {
        "baseline": ("Baseline", gpu_secp256k1_batch),
        "w4": ("Window4", gpu_secp256k1_batch_window4),
        "w6": ("Window6", gpu_secp256k1_batch_window6),
        "w6_glv": ("Window6-GLV", gpu_secp256k1_batch_window6_glv),
        "w8": ("Window8", gpu_secp256k1_batch_window8),
    }
    windows = _select_windows(args.windows, mapping)

    results: List[Dict[str, float]] = []
    for batch in batches:
        for label, func in windows:
            record = _measure_window(label, func, batch, repeats)
            results.append(record)
            print(
                f"[{label:10s}] batch={batch:6d} | "
                f"avg={record['avg_sec']*1000:8.2f} ms | "
                f"std={record['std_sec']*1000:6.2f} ms | "
                f"throughput={record['throughput']/1e6:7.3f} Mkeys/s"
            )

    if args.json:
        payload = {
            "batches": batches,
            "repeats": repeats,
            "windows": [label for label, _ in windows],
            "results": results,
            "module": args.module,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[info] Result json saved to: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
