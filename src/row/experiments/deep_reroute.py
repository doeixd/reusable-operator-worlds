"""Exhaustive route search that scales to depth 5 with bounded memory, and O5's re-route on top of it.

`split_route_mse(library, x, y, depth)`: support MSE of every hard route of length `depth`, flattened
lexicographically exactly as `FrozenLibrary.all_route_support_mse` (first position most significant). It computes
the states of all `slots**(depth-1)` prefixes in one pass (the existing all-route recursion), then applies the last
step to prefix chunks, so peak memory is the prefix tensor plus one chunk instead of `slots**depth` full states
(1.0 GiB per task at depth 5 and 64 examples, unchunked). Same candidate function, same arithmetic per route; the
argmin is required to equal `enum_route`'s at depths 2-4 (tests). `exhaustive_route` uses `enum_route` verbatim up to
`DIRECT_MAX_DEPTH` and the split search beyond it, so every committed depth-3/4 result is reproduced bitwise.

`reroute(model, stream_tasks, plan, pool)`: O5's construction verbatim (frozen library, every stream task, its
reservoir examples, minimal logit swap) with `exhaustive_route` as the chooser.
"""
from __future__ import annotations

import numpy as np
import torch

from row.experiments.audit_j2a_staged_library import enum_route
from row.experiments.audit_so1r_route_only import FrozenLibrary, unflatten

DIRECT_MAX_DEPTH = 4
PREFIX_CHUNK = 1024


def split_route_mse(library, x, y, depth, chunk=PREFIX_CHUNK):
    with torch.no_grad():
        n, d = x.shape
        z = x
        for _ in range(depth - 1):
            z = library.candidates(z.reshape(-1, d)).reshape(n, -1, d)   # (n, slots**t, d), lexicographic
        prefixes = z.shape[1]
        out = torch.empty(prefixes * library.slots)
        for start in range(0, prefixes, chunk):
            stop = min(start + chunk, prefixes)
            block = z[:, start:stop, :]                                      # (n, c, d)
            last = library.candidates(block.reshape(-1, d)).reshape(n, stop - start, library.slots, d)
            mse = torch.mean((last - y[:, None, None, :]) ** 2, dim=(0, 3))  # (c, slots)
            out[start * library.slots: stop * library.slots] = mse.reshape(-1)
        return out


def exhaustive_route(library, x, y, depth):
    library.steps = depth
    if depth <= DIRECT_MAX_DEPTH:
        return [int(r) for r in enum_route(library, x, y)]
    return [int(r) for r in unflatten(int(torch.argmin(split_route_mse(library, x, y, depth))), library.slots, depth)]


def reroute(model, stream_tasks, plan, pool):
    """O5's reroute with `exhaustive_route` as the chooser. Returns the number of tasks whose route changed."""
    library = FrozenLibrary(model)
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    current = model.hard_routes()
    changed = 0
    with torch.no_grad():
        for task in stream_tasks:
            d = plan[task.task_id]
            xs = torch.tensor(np.stack([a for a, _ in by[task.task_id]]), dtype=torch.float32)
            ys = torch.tensor(np.stack([b for _, b in by[task.task_id]]), dtype=torch.float32)
            new = exhaustive_route(library, xs, ys, d)
            code = model.task_codes[task.task_id]
            for step in range(d):
                old = int(torch.argmax(code[step]))
                if old != new[step]:
                    a, b = code[step, old].clone(), code[step, new[step]].clone()
                    code[step, old], code[step, new[step]] = b, a
            if [int(r) for r in current[task.task_id][:d]] != new:
                changed += 1
            if [int(r) for r in model.hard_routes()[task.task_id][:d]] != new:
                raise RuntimeError(f'{task.task_id}: route swap did not take')
    return changed
