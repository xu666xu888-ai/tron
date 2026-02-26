# -*- coding: utf-8 -*-
"""
Turbo Vanity Search Engine — Full End-to-End GPU Pipeline

Entire pipeline runs on GPU:
  secp256k1 → affine → Keccak-256 → SHA-256d → Base58 suffix match

Only matched keys are returned to CPU. Zero CPU-side address computation.

Supports:
  - Base58 suffix matching (fully on GPU)
  - Multi-GPU parallelization via multiprocessing
  - Rich progress display
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import secrets
import time
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


def _search_single_gpu(
    suffix: str,
    gpu_id: int,
    timeout: Optional[float],
    max_attempts: Optional[int],
    total_threads: int,
    iters_per_thread: int,
    progress_callback: Optional[Callable],
) -> dict:
    """Single-GPU turbo search — everything on GPU."""
    import cupy as cp
    cp.cuda.Device(gpu_id).use()

    from tron_vanity.gpu_incremental import gpu_vanity_scan, BATCH_K

    total_per_batch = total_threads * iters_per_thread * BATCH_K
    total_keys = 0
    start_time = time.perf_counter()

    # Warmup (compile kernel)
    gpu_vanity_scan(secrets.token_bytes(32), suffix=suffix, total_threads=1024, iters_per_thread=1)
    cp.cuda.Device().synchronize()

    while True:
        base_key = secrets.token_bytes(32)

        hits, batch_keys = gpu_vanity_scan(
            base_key,
            suffix=suffix,
            total_threads=total_threads,
            iters_per_thread=iters_per_thread,
        )

        total_keys += batch_keys
        elapsed = time.perf_counter() - start_time
        mkeys_per_sec = total_keys / elapsed / 1e6

        if progress_callback:
            progress_callback(total_keys, elapsed, mkeys_per_sec)

        # Hits already verified by GPU — Base58 suffix match done on-device
        if hits:
            return {
                "found": True,
                "privkey_hex": hits[0][0].hex(),
                "address_hex": hits[0][1],
                "address_base58": hits[0][2],
                "total_keys": total_keys,
                "elapsed": elapsed,
                "mkeys_per_sec": mkeys_per_sec,
                "gpu_id": gpu_id,
                "num_gpus": 1,
                "all_hits": hits,
            }

        # Check limits
        if timeout and elapsed >= timeout:
            return {
                "found": False, "reason": "timeout",
                "total_keys": total_keys, "elapsed": elapsed,
                "mkeys_per_sec": mkeys_per_sec, "num_gpus": 1,
            }
        if max_attempts and total_keys >= max_attempts:
            return {
                "found": False, "reason": "max_attempts",
                "total_keys": total_keys, "elapsed": elapsed,
                "mkeys_per_sec": mkeys_per_sec, "num_gpus": 1,
            }


def _gpu_worker(
    gpu_id: int,
    suffix: str,
    result_queue: mp.Queue,
    stop_event: mp.Event,
    stats_queue: mp.Queue,
    total_threads: int,
    iters_per_thread: int,
):
    """Worker function for multi-GPU mode."""
    try:
        import cupy as cp
        cp.cuda.Device(gpu_id).use()

        from tron_vanity.gpu_incremental import gpu_vanity_scan, BATCH_K

        total_per_batch = total_threads * iters_per_thread * BATCH_K

        # Warmup
        gpu_vanity_scan(secrets.token_bytes(32), suffix=suffix, total_threads=1024, iters_per_thread=1)
        cp.cuda.Device().synchronize()

        while not stop_event.is_set():
            base_key = secrets.token_bytes(32)
            hits, _ = gpu_vanity_scan(
                base_key, suffix=suffix,
                total_threads=total_threads, iters_per_thread=iters_per_thread,
            )

            stats_queue.put((gpu_id, total_per_batch))

            if hits:
                for pk_bytes, addr_hex, addr_b58 in hits:
                    result_queue.put((pk_bytes, addr_hex, addr_b58, gpu_id))
                return

    except Exception as e:
        logger.error(f"GPU {gpu_id} worker error: {e}", exc_info=True)
        stats_queue.put((gpu_id, 0, str(e)))


def _search_multi_gpu(
    suffix: str,
    gpu_ids: List[int],
    timeout: Optional[float],
    max_attempts: Optional[int],
    total_threads: int,
    iters_per_thread: int,
    progress_callback: Optional[Callable],
) -> dict:
    """Multi-GPU turbo search using multiprocessing."""
    result_queue = mp.Queue()
    stats_queue = mp.Queue()
    stop_event = mp.Event()

    workers = []
    for gpu_id in gpu_ids:
        p = mp.Process(
            target=_gpu_worker,
            args=(gpu_id, suffix, result_queue, stop_event, stats_queue,
                  total_threads, iters_per_thread),
            daemon=True,
        )
        p.start()
        workers.append(p)

    start_time = time.perf_counter()
    total_keys = 0

    try:
        while True:
            if not result_queue.empty():
                pk_bytes, addr_hex, addr_b58, gpu_id = result_queue.get_nowait()
                elapsed = time.perf_counter() - start_time
                stop_event.set()
                return {
                    "found": True,
                    "privkey_hex": pk_bytes.hex(),
                    "address_hex": addr_hex,
                    "address_base58": addr_b58,
                    "total_keys": total_keys,
                    "elapsed": elapsed,
                    "mkeys_per_sec": total_keys / max(elapsed, 0.001) / 1e6,
                    "gpu_id": gpu_id,
                    "num_gpus": len(gpu_ids),
                }

            while not stats_queue.empty():
                data = stats_queue.get_nowait()
                if len(data) == 2:
                    gid, keys = data
                    total_keys += keys

            elapsed = time.perf_counter() - start_time
            mkeys_per_sec = total_keys / max(elapsed, 0.001) / 1e6

            if progress_callback:
                progress_callback(total_keys, elapsed, mkeys_per_sec)

            if timeout and elapsed >= timeout:
                stop_event.set()
                return {
                    "found": False, "reason": "timeout",
                    "total_keys": total_keys, "elapsed": elapsed,
                    "mkeys_per_sec": mkeys_per_sec, "num_gpus": len(gpu_ids),
                }
            if max_attempts and total_keys >= max_attempts:
                stop_event.set()
                return {
                    "found": False, "reason": "max_attempts",
                    "total_keys": total_keys, "elapsed": elapsed,
                    "mkeys_per_sec": mkeys_per_sec, "num_gpus": len(gpu_ids),
                }

            alive = [p for p in workers if p.is_alive()]
            if not alive:
                elapsed = time.perf_counter() - start_time
                return {
                    "found": False, "reason": "workers_died",
                    "total_keys": total_keys, "elapsed": elapsed,
                    "mkeys_per_sec": total_keys / max(elapsed, 0.001) / 1e6,
                    "num_gpus": len(gpu_ids),
                }

            time.sleep(0.05)

    finally:
        stop_event.set()
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def search_vanity_turbo(
    suffix: str,
    timeout: Optional[float] = None,
    max_attempts: Optional[int] = None,
    gpu_ids: Optional[List[int]] = None,
    total_threads: int = 32768,
    iters_per_thread: int = 2,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Run turbo vanity search — full end-to-end on GPU."""
    import cupy as cp

    if gpu_ids is None:
        num_gpus = cp.cuda.runtime.getDeviceCount()
        gpu_ids = list(range(num_gpus))

    logger.info(f"Turbo search: suffix='{suffix}', GPUs={gpu_ids}")

    if len(gpu_ids) == 1:
        return _search_single_gpu(
            suffix, gpu_ids[0], timeout, max_attempts,
            total_threads, iters_per_thread, progress_callback,
        )
    else:
        return _search_multi_gpu(
            suffix, gpu_ids, timeout, max_attempts,
            total_threads, iters_per_thread, progress_callback,
        )


def cli_turbo_search(
    suffix: str,
    timeout: Optional[float] = None,
    max_attempts: Optional[int] = None,
    gpu_ids: Optional[List[int]] = None,
) -> int:
    """CLI entry point with Rich progress display."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
    except ImportError:
        console = None

    import cupy as cp
    if gpu_ids is None:
        num_gpus = cp.cuda.runtime.getDeviceCount()
        gpu_ids = list(range(num_gpus))

    if console:
        console.print(f"\n[bold cyan]🚀 TRON Turbo Vanity Search[/bold cyan]")
        console.print(f"  Target suffix: [bold yellow]{suffix}[/bold yellow]")
        console.print(f"  GPUs: [green]{len(gpu_ids)}[/green] ({', '.join(f'GPU:{g}' for g in gpu_ids)})")
        console.print(f"  Engine: [green]Full GPU pipeline (secp256k1→Keccak→SHA256→Base58)[/green]")
        if timeout:
            console.print(f"  Timeout: {timeout}s")
        console.print()

    last_print = [0.0]

    def progress_cb(total_keys, elapsed, mkeys):
        now = time.time()
        if now - last_print[0] < 1.0:
            return
        last_print[0] = now
        if console:
            console.print(
                f"  [cyan]{elapsed:7.1f}s[/cyan] | "
                f"[green]{mkeys:7.2f} Mkeys/s[/green] | "
                f"Scanned: [white]{total_keys/1e9:.3f}B[/white] keys",
                end="\r",
            )
        else:
            print(f"  {elapsed:7.1f}s | {mkeys:7.2f} Mkeys/s | {total_keys/1e9:.3f}B keys", end="\r")

    result = search_vanity_turbo(
        suffix=suffix,
        timeout=timeout,
        max_attempts=max_attempts,
        gpu_ids=gpu_ids,
        progress_callback=progress_cb,
    )

    if console:
        console.print()
        if result["found"]:
            summary = (
                f"[bold green]🎉 Found vanity address![/bold green]\n\n"
                f"  Base58: [bold white]{result['address_base58']}[/bold white]\n"
                f"  Hex:    {result['address_hex']}\n"
                f"  Privkey: [bold red]{result['privkey_hex']}[/bold red]\n\n"
                f"  Scanned: {result['total_keys']/1e6:.1f}M keys in {result['elapsed']:.2f}s\n"
                f"  Speed:   {result['mkeys_per_sec']:.2f} Mkeys/s "
                f"({result['num_gpus']} GPU{'s' if result['num_gpus']>1 else ''})"
            )
            console.print(Panel(summary, border_style="green", title="✅ SUCCESS"))
        else:
            console.print(f"[yellow]Not found. Reason: {result.get('reason', 'unknown')}[/yellow]")
            console.print(f"  Scanned: {result['total_keys']/1e6:.1f}M keys in {result['elapsed']:.2f}s")
            console.print(f"  Speed:   {result.get('mkeys_per_sec', 0):.2f} Mkeys/s")
    else:
        print()
        if result["found"]:
            print(f"FOUND: {result['address_base58']}")
            print(f"Privkey: {result['privkey_hex']}")
        else:
            print(f"Not found: {result.get('reason')}")

    return 0 if result["found"] else 1
