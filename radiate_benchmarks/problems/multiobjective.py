"""Multi-objective benchmarks: ZDT1, ZDT3, DTLZ2.

Same single-vector / batched pairing as ``continuous.py``. ``ref_point`` is
a fixed reference point used for hypervolume tracking so every library's
Pareto front is scored on the same scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MOProblem:
    name: str
    n_var: int
    n_obj: int
    bounds: tuple[float, float]
    fn: Callable[[np.ndarray], tuple[float, ...]]
    batch_fn: Callable[[np.ndarray], np.ndarray]
    ref_point: tuple[float, ...]


def _zdt1(x: np.ndarray) -> tuple[float, ...]:
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    h = 1.0 - np.sqrt(f1 / g)
    return (float(f1), float(g * h))


def _zdt1_batch(X: np.ndarray) -> np.ndarray:
    f1 = X[:, 0]
    g = 1.0 + 9.0 * np.sum(X[:, 1:], axis=1) / (X.shape[1] - 1)
    h = 1.0 - np.sqrt(f1 / g)
    return np.stack([f1, g * h], axis=1)


def _zdt3(x: np.ndarray) -> tuple[float, ...]:
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
    return (float(f1), float(g * h))


def _zdt3_batch(X: np.ndarray) -> np.ndarray:
    f1 = X[:, 0]
    g = 1.0 + 9.0 * np.sum(X[:, 1:], axis=1) / (X.shape[1] - 1)
    h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
    return np.stack([f1, g * h], axis=1)


DTLZ2_N_OBJ = 3
DTLZ2_K = 10
DTLZ2_N_VAR = DTLZ2_N_OBJ + DTLZ2_K - 1


def _dtlz2(x: np.ndarray) -> tuple[float, ...]:
    m = DTLZ2_N_OBJ
    xm = x[m - 1 :]
    g = float(np.sum((xm - 0.5) ** 2))
    f = []
    for i in range(m):
        val = 1.0 + g
        for j in range(m - 1 - i):
            val *= np.cos(x[j] * np.pi / 2.0)
        if i > 0:
            val *= np.sin(x[m - 1 - i] * np.pi / 2.0)
        f.append(float(val))
    return tuple(f)


def _dtlz2_batch(X: np.ndarray) -> np.ndarray:
    return np.array([_dtlz2(row) for row in X])


PROBLEMS: list[MOProblem] = [
    MOProblem("zdt1", 30, 2, (0.0, 1.0), _zdt1, _zdt1_batch, (1.1, 1.1)),
    MOProblem("zdt3", 30, 2, (0.0, 1.0), _zdt3, _zdt3_batch, (1.1, 1.1)),
    MOProblem(
        "dtlz2",
        DTLZ2_N_VAR,
        DTLZ2_N_OBJ,
        (0.0, 1.0),
        _dtlz2,
        _dtlz2_batch,
        (1.1, 1.1, 1.1),
    ),
]
