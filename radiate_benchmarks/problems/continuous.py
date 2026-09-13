"""Classic continuous function-optimization benchmarks.

Every problem exposes a single-vector evaluator ``fn(x) -> float`` (used by
DEAP/PyGAD/Radiate, which evaluate one individual at a time) and a batched
``batch_fn(X) -> np.ndarray`` (used by pymoo, which vectorizes over the
whole population). Both compute the exact same formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ContinuousProblem:
    name: str
    dim: int
    bounds: tuple[float, float]
    fn: Callable[[np.ndarray], float]
    batch_fn: Callable[[np.ndarray], np.ndarray]
    optimum: float  # known global minimum value


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def sphere_batch(X: np.ndarray) -> np.ndarray:
    return np.sum(X * X, axis=1)


def rastrigin(x: np.ndarray) -> float:
    return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def rastrigin_batch(X: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    return 10 * n + np.sum(X * X - 10 * np.cos(2 * np.pi * X), axis=1)


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rosenbrock_batch(X: np.ndarray) -> np.ndarray:
    return np.sum(
        100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2, axis=1
    )


def ackley(x: np.ndarray) -> float:
    n = len(x)
    sum1 = np.sum(x * x)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return float(
        -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
        - np.exp(sum2 / n)
        + 20
        + np.e
    )


def ackley_batch(X: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    sum1 = np.sum(X * X, axis=1)
    sum2 = np.sum(np.cos(2 * np.pi * X), axis=1)
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


DIM = 30

PROBLEMS: list[ContinuousProblem] = [
    ContinuousProblem("sphere", DIM, (-5.12, 5.12), sphere, sphere_batch, 0.0),
    ContinuousProblem(
        "rastrigin", DIM, (-5.12, 5.12), rastrigin, rastrigin_batch, 0.0
    ),
    ContinuousProblem(
        "rosenbrock", DIM, (-2.048, 2.048), rosenbrock, rosenbrock_batch, 0.0
    ),
    ContinuousProblem("ackley", DIM, (-32.768, 32.768), ackley, ackley_batch, 0.0),
]
