"""Discrete/combinatorial benchmarks: 0/1 Knapsack, TSP, N-Queens.

Problem instances are generated once with a fixed instance seed so every
library solves the exact same instance; the *trial* seed (passed separately
by the harness) only controls the EA's own randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

INSTANCE_SEED = 42


# ---------------------------------------------------------------- Knapsack
@dataclass(frozen=True)
class KnapsackProblem:
    name: str
    n_items: int
    weights: np.ndarray
    values: np.ndarray
    capacity: float
    fn: Callable[[np.ndarray], float]  # maximize


def _make_knapsack(n_items: int = 50) -> KnapsackProblem:
    rng = np.random.default_rng(INSTANCE_SEED)
    weights = rng.integers(1, 50, size=n_items).astype(float)
    values = rng.integers(1, 100, size=n_items).astype(float)
    capacity = 0.5 * weights.sum()

    def fitness(bits: np.ndarray) -> float:
        bits = np.asarray(bits)
        total_weight = float(np.sum(weights * bits))
        total_value = float(np.sum(values * bits))
        if total_weight > capacity:
            return total_value - 2.0 * (total_weight - capacity)
        return total_value

    return KnapsackProblem("knapsack", n_items, weights, values, capacity, fitness)


KNAPSACK = _make_knapsack()


# --------------------------------------------------------------------- TSP
@dataclass(frozen=True)
class TSPProblem:
    name: str
    n_cities: int
    coords: np.ndarray  # (n_cities, 2)
    dist: np.ndarray  # (n_cities, n_cities) precomputed distance matrix
    fn: Callable[[list[int]], float]  # minimize


def _make_tsp(n_cities: int = 30) -> TSPProblem:
    rng = np.random.default_rng(INSTANCE_SEED)
    coords = rng.uniform(0, 100, size=(n_cities, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))

    def fitness(tour: list[int]) -> float:
        idx = np.asarray(tour)
        nxt = np.roll(idx, -1)
        return float(np.sum(dist[idx, nxt]))

    return TSPProblem("tsp", n_cities, coords, dist, fitness)


TSP = _make_tsp()


# ---------------------------------------------------------------- N-Queens
@dataclass(frozen=True)
class NQueensProblem:
    name: str
    n: int
    fn: Callable[[list[int]], float]  # minimize (0 == solved)


def _nqueens_conflicts(perm: list[int]) -> float:
    n = len(perm)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(perm[i] - perm[j]) == abs(i - j):
                conflicts += 1
    return float(conflicts)


NQUEENS = NQueensProblem("nqueens", 20, _nqueens_conflicts)
