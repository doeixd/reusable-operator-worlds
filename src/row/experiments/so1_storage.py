"""Durable SO1 cells and explicit provenance; no scientific decision rules."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=1, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


@contextmanager
def writer_lock(path):
    """OS lock released on process death; the lock file may safely remain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"another writer owns {path}") from error
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def resolved(config):
    value = json.loads(json.dumps(asdict(config), default=str))
    value["output"] = {"directory": value.pop("output_directory")}
    return value


def environment():
    return {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "psutil": psutil.__version__, "platform": platform.platform(),
            "torch_threads": torch.get_num_threads()}


def memory_snapshot():
    vm = psutil.virtual_memory()
    # Installed psutil's Windows implementation exposes GlobalMemoryStatusEx
    # total/available system commitment here; swap.free is NOT commit headroom.
    if os.name == "nt":
        _, _, limit, available = psutil._psplatform.cext.virtual_mem()
    else:
        raise RuntimeError("SO1 host calibration currently requires Windows")
    own = psutil.Process().memory_info()
    return {"utc": now(), "physical_available": vm.available,
            "commit_limit": limit, "commit_available": available,
            "pagefile_used": psutil.swap_memory().used,
            "rss": own.rss, "private": own.private,
            "peak_rss": own.peak_wset, "peak_private": own.peak_pagefile}


class MemorySampler:
    """Sample this process plus its children, including interpreter startup."""
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.errors = []

    def sample(self):
        try:
            sample = memory_snapshot()
            children = []
            for proc in psutil.Process().children(recursive=True):
                try:
                    info = proc.memory_info()
                    children.append({"pid": proc.pid, "rss": info.rss,
                                     "private": info.private, "peak_rss": info.peak_wset,
                                     "peak_private": info.peak_pagefile})
                except psutil.NoSuchProcess:
                    pass
            sample["children"] = children
            self.samples.append(sample)
        except Exception as error:
            self.errors.append(repr(error))

    def __enter__(self):
        self.sample()
        def loop():
            while not self.stop.wait(0.2):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()

    def summary(self):
        children = [p for s in self.samples for p in s["children"]]
        return {"samples": len(self.samples), "errors": self.errors,
                "min_physical_available": min(s["physical_available"] for s in self.samples),
                "min_commit_available": min(s["commit_available"] for s in self.samples),
                "max_pagefile_used": max(s["pagefile_used"] for s in self.samples),
                "pagefile_growth": max(s["pagefile_used"] for s in self.samples) - self.samples[0]["pagefile_used"],
                "max_worker_rss": max((p["peak_rss"] for p in children), default=0),
                "max_worker_private": max((p["peak_private"] for p in children), default=0),
                "max_parent_rss": max(s["peak_rss"] for s in self.samples),
                "max_parent_private": max(s["peak_private"] for s in self.samples)}


def world_digest(world):
    h = hashlib.sha256()
    for task in world.tasks:
        h.update(task.task_id.encode())
        for data in (task.train_x, task.train_y, task.eval_x, task.eval_y):
            array = np.ascontiguousarray(data)
            h.update(str(array.shape).encode())
            h.update(array.dtype.str.encode())
            h.update(array.tobytes())
    return h.hexdigest()


def cell_stamp(job):
    return {k: job[k] for k in ("key", "world", "oracle", "updates", "batch",
                                "cell_index", "sampling_index", "git_commit", "protocol_sha256")}


def load_cell(path, expected_stamp):
    """Refuse incomplete/corrupt/cross-protocol results, never silently reuse."""
    path = Path(path)
    result = json.loads((path / "result.json").read_text())
    if result["stamp"] != expected_stamp or result.get("complete") is not True:
        raise ValueError(f"cell provenance mismatch: {path}")
    for name, sha in result["artifact_sha256"].items():
        if digest(path / name) != sha:
            raise ValueError(f"cell artifact hash mismatch: {path / name}")
    if result["result_sha256"] != fingerprint(result["result"]):
        raise ValueError(f"cell result hash mismatch: {path}")
    return result["result"]


def save_model(path, model, cfg):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt.tmp")
    os.replace(path / "model.pt.tmp", path / "model.pt")
    state = {"temperature": model.temperature, "training": model.training,
             "requires_grad": {k: p.requires_grad for k, p in model.named_parameters()}}
    atomic_json(path / "model_state.json", state)
    (path / "config.yaml").write_text(yaml.safe_dump(resolved(cfg)), encoding="utf-8")


def restore_model(path, cfg, world, builder):
    path = Path(path)
    model = builder(cfg)
    for task in world.tasks:
        model.begin_task(task.task_id)
    model.load_state_dict(torch.load(path / "model.pt", weights_only=True), strict=True)
    state = json.loads((path / "model_state.json").read_text())
    if set(state["requires_grad"]) != set(dict(model.named_parameters())):
        raise ValueError("incomplete requires_grad state")
    for name, param in model.named_parameters():
        param.requires_grad_(state["requires_grad"][name])
    model.temperature = state["temperature"]
    model.train(state["training"])
    return model
