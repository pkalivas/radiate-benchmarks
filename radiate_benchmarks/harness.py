"""Wires problems x libraries x seeds together and runs the trials."""

from __future__ import annotations

from types import ModuleType

from radiate_benchmarks.adapters import (
    deap_adapter,
    # pygad_adapter,
    pymoo_adapter,
    radiate_adapter,
)
from radiate_benchmarks.adapters.base import BenchmarkResult, Config
from radiate_benchmarks.problems.combinatorial import KNAPSACK, NQUEENS, TSP
from radiate_benchmarks.problems.continuous import PROBLEMS as CONTINUOUS_PROBLEMS
from radiate_benchmarks.problems.multiobjective import PROBLEMS as MO_PROBLEMS

LIBRARIES = {
    "radiate": radiate_adapter,
    "deap": deap_adapter,
    "pymoo": pymoo_adapter,
    # "pygad": pygad_adapter,
}

SUITES = {
    "continuous": [(p, "run_continuous") for p in CONTINUOUS_PROBLEMS],
    "combinatorial": [
        (KNAPSACK, "run_knapsack"),
        (TSP, "run_tsp"),
        (NQUEENS, "run_nqueens"),
    ],
    "multiobjective": [(p, "run_mo") for p in MO_PROBLEMS],
}


def run_suite(
    suite_name: str,
    config: Config,
    seeds: list[int],
    libraries: dict[str, ModuleType] = LIBRARIES,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    entries = SUITES[suite_name]

    for problem, method_name in entries:
        for lib_name, module in libraries.items():
            fn = getattr(module, method_name)
            for seed in seeds:
                print(
                    f"[{suite_name}] {problem.name:>12} | {lib_name:>8} | seed={seed}",
                    end=" ",
                )
                try:
                    result = fn(problem, config, seed)
                except Exception as exc:  # noqa: BLE001 - keep the benchmark run going
                    print(f"FAILED: {exc}")
                    continue
                print(
                    f"-> best={result.best_fitness:.4f}  time={result.wall_time_s:.3f}s"
                )
                results.append(result)

    return results


def run_all(
    config: Config, seeds: list[int], libraries: dict[str, ModuleType] = LIBRARIES
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for suite_name in SUITES:
        results.extend(run_suite(suite_name, config, seeds, libraries))
    return results
