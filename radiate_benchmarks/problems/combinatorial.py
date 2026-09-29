"""Discrete/combinatorial benchmarks: 0/1 Knapsack, TSP, N-Queens.

Problem instances are generated once with a fixed instance seed and written into
the spec, so every language and library solves the exact same instance; the
*trial* seed only controls the EA's own randomness. Every other consumer loads
the instance back from the spec JSON via ``*_from_spec`` rather than regenerating it.
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


def knapsack_spec(n_items: int = 50) -> dict:
    rng = np.random.default_rng(INSTANCE_SEED)
    weights = rng.integers(1, 50, size=n_items).astype(float)
    values = rng.integers(1, 100, size=n_items).astype(float)
    return {
        "name": "knapsack",
        "suite": "combinatorial",
        "kind": "knapsack",
        "n_items": n_items,
        "weights": weights.tolist(),
        "values": values.tolist(),
        "capacity": float(0.5 * weights.sum()),
        "overweight_penalty": 2.0,
        "minimize": False,
    }


def knapsack_from_spec(spec: dict) -> KnapsackProblem:
    weights = np.asarray(spec["weights"], dtype=float)
    values = np.asarray(spec["values"], dtype=float)
    capacity = float(spec["capacity"])
    penalty = float(spec["overweight_penalty"])

    def fitness(bits: np.ndarray) -> float:
        bits = np.asarray(bits)
        total_weight = float(np.sum(weights * bits))
        total_value = float(np.sum(values * bits))
        if total_weight > capacity:
            return total_value - penalty * (total_weight - capacity)
        return total_value

    return KnapsackProblem(
        spec["name"], spec["n_items"], weights, values, capacity, fitness
    )


# --------------------------------------------------------------------- TSP
@dataclass(frozen=True)
class TSPProblem:
    name: str
    n_cities: int
    coords: np.ndarray  # (n_cities, 2)
    dist: np.ndarray  # (n_cities, n_cities) precomputed distance matrix
    fn: Callable[
        [list[int]], float
    ]  # minimize, closed tour (returns to the start city)


def tsp_spec(n_cities: int = 30) -> dict:
    rng = np.random.default_rng(INSTANCE_SEED)
    coords = rng.uniform(0, 100, size=(n_cities, 2))
    return {
        "name": "tsp",
        "suite": "combinatorial",
        "kind": "tsp",
        "n_cities": n_cities,
        "coords": coords.tolist(),
        "minimize": True,
    }


def tsp_from_spec(spec: dict) -> TSPProblem:
    coords = np.asarray(spec["coords"], dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))

    def fitness(tour: list[int]) -> float:
        idx = np.asarray(tour)
        nxt = np.roll(idx, -1)
        return float(np.sum(dist[idx, nxt]))

    return TSPProblem(spec["name"], spec["n_cities"], coords, dist, fitness)


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


def nqueens_spec(n: int = 20) -> dict:
    return {
        "name": "nqueens",
        "suite": "combinatorial",
        "kind": "nqueens",
        "n": n,
        "minimize": True,
    }


def nqueens_from_spec(spec: dict) -> NQueensProblem:
    return NQueensProblem(spec["name"], spec["n"], _nqueens_conflicts)


def spec_entries() -> list[dict]:
    return [knapsack_spec(), tsp_spec(), nqueens_spec()]
