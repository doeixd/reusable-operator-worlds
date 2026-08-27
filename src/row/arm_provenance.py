"""Arm fingerprints: make `S != S` a machine-checkable fact.

Review 82, after E5's scratch arm was found to be a FINE-TUNING arm wearing the
scratch label. E1 and E8 built it with `scratch_model(...)`; E5 built it with
`copy.deepcopy(trained_model)`. Both call sites read `adapt_cell(..., S, ...)`
and nothing in the artifact recorded the difference.

An arm is a CONSTRUCTION, not a name. `describe_arm` records what a construction
actually was, so a registered experiment can assert it instead of trusting the
label, and so two arms sharing a name can be compared and found different.

Typical use, in a scorer::

    record = describe_arm("S", model, init_source="fresh", steps=2000,
                          trainable=params, optimizer=optimizer)
    assert_arm(record, init_source="fresh", trainable_count=EXPECTED)

The parameter hash is over INITIAL state, before any adaptation, because that is
what distinguishes a fresh model from a trained one; two arms that trained to the
same place from different starts are still different arms.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import torch


def tensor_digest(model: torch.nn.Module) -> str:
    """A stable hash of every parameter, in sorted-name order."""
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()[:16]


def describe_arm(name: str, model: torch.nn.Module, *, init_source: str,
                 steps: int, trainable: Iterable[torch.Tensor] | None = None,
                 optimizer: torch.optim.Optimizer | None = None,
                 fresh_seed: int | None = None,
                 data_seen: str | None = None) -> dict[str, Any]:
    """Record what this arm actually IS, before it is adapted.

    `init_source` is the one field that caught E5: use "fresh" for a newly
    constructed model, "trained" for one restored from an artifact, and
    "copy:<origin>" for a deep copy of another arm's model.
    """
    if init_source not in ("fresh", "trained") and not init_source.startswith("copy:"):
        raise ValueError(f"init_source must be fresh|trained|copy:<origin>, got {init_source!r}")
    trainable = list(trainable) if trainable is not None else [
        p for p in model.parameters() if p.requires_grad]
    trainable_ids = {id(p) for p in trainable}
    frozen = [p for p in model.parameters() if id(p) not in trainable_ids]
    return {
        "arm_name": name,
        "init_source": init_source,
        "checkpoint_hash": tensor_digest(model),
        "fresh_seed": fresh_seed,
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
        "trainable_tensors": len(trainable),
        "frozen_parameters": int(sum(p.numel() for p in frozen)),
        "optimizer": type(optimizer).__name__ if optimizer is not None else None,
        "optimizer_lr": (float(optimizer.param_groups[0]["lr"])
                         if optimizer is not None and optimizer.param_groups else None),
        "steps": int(steps),
        "data_seen": data_seen,
    }


def assert_arm(record: dict[str, Any], **expected: Any) -> None:
    """Fail closed when an arm is not the construction the plan registered."""
    for field, want in expected.items():
        got = record.get(field)
        if got != want:
            raise SystemExit(
                f"FATAL: arm {record.get('arm_name')!r} field {field} is {got!r}, "
                f"registered as {want!r}. An arm is a construction, not a name.")


def arms_differ(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Fields on which two same-named arms disagree; empty means comparable."""
    fields = ("init_source", "trainable_parameters", "frozen_parameters",
              "optimizer", "optimizer_lr", "steps")
    return [f for f in fields if a.get(f) != b.get(f)]


def digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]
