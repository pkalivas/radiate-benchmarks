"""Reference problem definitions, shared by the spec generator, the Python runner and
the aggregator's re-scoring step.

A problem travels between languages as a JSON dict (see ``spec_entries`` and
``schema.md``); ``from_spec`` turns one of those dicts back into a Python problem
object with a callable fitness function.
"""

from __future__ import annotations

import numpy as np

from radiate_benchmarks.problems import combinatorial, continuous, multiobjective
from radiate_benchmarks.problems.combinatorial import (
    KnapsackProblem,
    NQueensProblem,
    TSPProblem,
)
from radiate_benchmarks.problems.continuous import ContinuousProblem
from radiate_benchmarks.problems.multiobjective import MOProblem

__all__ = [
    "ContinuousProblem",
    "KnapsackProblem",
    "MOProblem",
    "NQueensProblem",
    "TSPProblem",
    "from_spec",
    "reference_score",
    "spec_entries",
]

_FROM_SPEC = {
    "continuous": continuous.from_spec,
    "knapsack": combinatorial.knapsack_from_spec,
    "tsp": combinatorial.tsp_from_spec,
    "nqueens": combinatorial.nqueens_from_spec,
    "mo": multiobjective.from_spec,
}


def spec_entries() -> list[dict]:
    """Every problem, in suite order, as a JSON-serializable spec dict."""
    return [
        *continuous.spec_entries(),
        *combinatorial.spec_entries(),
        *multiobjective.spec_entries(),
    ]


def from_spec(spec: dict):
    return _FROM_SPEC[spec["kind"]](spec)


def reference_score(spec: dict, solution: list[float]) -> float:
    """Score a single-objective solution with the reference fitness function."""
    problem = from_spec(spec)
    kind = spec["kind"]
    if kind in ("continuous", "knapsack"):
        return float(problem.fn(np.asarray(solution, dtype=float)))
    if kind in ("tsp", "nqueens"):
        return float(problem.fn([int(g) for g in solution]))
    raise ValueError(f"no single-objective reference score for kind={kind!r}")
