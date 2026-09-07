"""Bounded process pool for independent one-writer cells (CONCURRENCY_PLAN.md).

Rules implemented here, all from the plan:

- cap = min(cores - 2, floor((free_memory - reserve) / measured_rss)), computed
  from a MEASURED resident size for the family being run, never a constant;
- the memory test is applied AT DISPATCH against live free memory, because
  free memory on this host swings by gigabytes with non-research processes;
- every worker pins one torch thread;
- a failing cell fails the batch with a nonzero exit rather than leaving a
  silently missing cell;
- nothing here writes results: the job function is the single writer for its
  own cell, and the driver only collects return values.

The bitwise serial-versus-pooled equivalence gate lives in
`tools/pool_equivalence_gate.py` and must pass before this pool is used for a
scientific batch of a new cell family.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

import psutil

DEFAULT_RESERVE_BYTES = 4 * 1024**3
CORE_HEADROOM = 2


def free_memory_bytes() -> int:
    return int(psutil.virtual_memory().available)


@dataclass(frozen=True)
class PoolBudget:
    measured_rss_bytes: int
    reserve_bytes: int = DEFAULT_RESERVE_BYTES
    cores: int = os.cpu_count() or 1
    hard_cap: int | None = None

    def cap(self, free_bytes: int) -> int:
        if self.measured_rss_bytes <= 0:
            raise ValueError("measured_rss_bytes must be a measured positive size")
        by_memory = (free_bytes - self.reserve_bytes) // self.measured_rss_bytes
        by_cores = max(1, self.cores - CORE_HEADROOM)
        cap = min(by_cores, by_memory)
        if self.hard_cap is not None:
            cap = min(cap, self.hard_cap)
        return max(0, int(cap))

    def can_dispatch(self, running: int, free_bytes: int) -> bool:
        """Start one more worker only if live free memory still covers it."""
        return running < self.cap(free_bytes)


def _worker(job_function: Callable[[Any], Any], job: Any) -> Any:
    import torch  # local import so the parent need not hold torch

    torch.set_num_threads(1)
    return job_function(job)


class BatchFailed(RuntimeError):
    pass


def run_pool(
    job_function: Callable[[Any], Any],
    jobs: Sequence[Any],
    budget: PoolBudget,
    *,
    free_probe: Callable[[], int] = free_memory_bytes,
    poll_seconds: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> list[Any]:
    """Run `job_function(job)` for every job, at most `cap` at once.

    Returns results in job order. Raises BatchFailed on the first failing
    job after the running jobs finish, so no cell can be silently missing.
    """
    results: list[Any] = [None] * len(jobs)
    pending = list(enumerate(jobs))
    running: dict[Future, int] = {}
    failures: list[tuple[int, BaseException]] = []
    initial_cap = budget.cap(free_probe())
    if initial_cap < 1:
        raise BatchFailed(
            f"no room for one worker: free {free_probe() / 2**30:.2f} GiB, "
            f"reserve {budget.reserve_bytes / 2**30:.2f} GiB, "
            f"rss {budget.measured_rss_bytes / 2**20:.0f} MiB"
        )
    log(f"[pool] cap {initial_cap} (cores {budget.cores}, rss "
        f"{budget.measured_rss_bytes / 2**20:.0f} MiB, free {free_probe() / 2**30:.2f} GiB)")
    with ProcessPoolExecutor(max_workers=max(1, min(initial_cap, len(jobs)))) as executor:
        while pending or running:
            while pending and budget.can_dispatch(len(running), free_probe()) and not failures:
                index, job = pending.pop(0)
                running[executor.submit(_worker, job_function, job)] = index
                log(f"[pool] dispatched job {index} ({len(running)} running)")
            if not running:
                if failures:
                    break
                sleep(poll_seconds)
                continue
            done, _ = wait(list(running), timeout=poll_seconds, return_when=FIRST_COMPLETED)
            for future in done:
                index = running.pop(future)
                try:
                    results[index] = future.result()
                    log(f"[pool] job {index} done")
                except BaseException as error:  # noqa: BLE001 - reported, then raised
                    failures.append((index, error))
                    log(f"[pool] job {index} FAILED: {error!r}")
            if failures and not running:
                break
    if failures:
        detail = "; ".join(f"job {i}: {e!r}" for i, e in failures)
        raise BatchFailed(f"{len(failures)} job(s) failed, {len(pending)} never started: {detail}")
    return results


def measure_rss_bytes(pid: int | None = None) -> int:
    """Resident size of a running process, for the family being launched."""
    return int(psutil.Process(pid).memory_info().rss)
