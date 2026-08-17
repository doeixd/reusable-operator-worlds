"""Metrics used by lifetime and control experiments."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def mse(prediction: Array, target: Array) -> float:
    return float(np.mean(np.square(np.asarray(target) - np.asarray(prediction))))


def nmse(prediction: Array, target: Array) -> float:
    target = np.asarray(target, dtype=np.float64)
    denominator = float(np.mean(np.square(target - np.mean(target, axis=0, keepdims=True))))
    if denominator <= 0.0:
        raise ValueError("NMSE is undefined for a constant target")
    return mse(prediction, target) / denominator


def gaussian_nll(prediction: Array, target: Array, sigma: float = 0.1) -> float:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    error = np.asarray(target) - np.asarray(prediction)
    per_scalar = 0.5 * np.square(error / sigma) + math.log(sigma) + 0.5 * math.log(2.0 * math.pi)
    return float(np.sum(per_scalar))


def examples_to_criterion(curve: dict[int, float], threshold: float, censor_at: int) -> int:
    for n in sorted(curve):
        if curve[n] < threshold:
            return n
    return censor_at + 1

