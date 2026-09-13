from __future__ import annotations

import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import (
    BinaryRandomSampling,
    FloatRandomSampling,
    PermutationRandomSampling,
)
from pymoo.optimize import minimize

from radiate_benchmarks.adapters.base import BenchmarkResult, Config, hypervolume
from radiate_benchmarks.problems.combinatorial import (
    KnapsackProblem,
    NQueensProblem,
    TSPProblem,
)
from radiate_benchmarks.problems.continuous import ContinuousProblem
from radiate_benchmarks.problems.multiobjective import MOProblem

LIBRARY = "pymoo"


def run_continuous(problem: ContinuousProblem, config: Config, seed: int) -> BenchmarkResult:
    low, high = problem.bounds

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=problem.dim, n_obj=1, xl=low, xu=high)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = problem.batch_fn(X)

    algorithm = GA(
        pop_size=config.population_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=config.crossover_rate, eta=15),
        mutation=PM(prob=config.mutation_rate, eta=20),
        eliminate_duplicates=True,
    )

    t0 = time.perf_counter()
    res = minimize(
        _P(),
        algorithm,
        ("n_gen", config.generations),
        seed=seed,
        save_history=True,
        verbose=False,
    )
    wall = time.perf_counter() - t0

    history = [float(np.min(h.opt.get("F"))) for h in res.history]
    best = float(np.min(res.F))
    return BenchmarkResult(LIBRARY, problem.name, seed, best, history, wall)


def run_knapsack(problem: KnapsackProblem, config: Config, seed: int) -> BenchmarkResult:
    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=problem.n_items, n_obj=1, xl=0, xu=1, vtype=bool)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = -np.array([problem.fn(row.astype(float)) for row in X])

    algorithm = GA(
        pop_size=config.population_size,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=config.crossover_rate),
        mutation=BitflipMutation(prob=config.mutation_rate),
        eliminate_duplicates=True,
    )

    t0 = time.perf_counter()
    res = minimize(
        _P(),
        algorithm,
        ("n_gen", config.generations),
        seed=seed,
        save_history=True,
        verbose=False,
    )
    wall = time.perf_counter() - t0

    history = [float(-np.min(h.opt.get("F"))) for h in res.history]
    best = float(-np.min(res.F))
    return BenchmarkResult(
        LIBRARY, problem.name, seed, best, history, wall, minimize=False
    )


def _run_permutation(name: str, n: int, fitness_fn, config: Config, seed: int) -> BenchmarkResult:
    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=1, xl=0, xu=n - 1, vtype=int)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = np.array([fitness_fn(list(row)) for row in X])

    algorithm = GA(
        pop_size=config.population_size,
        sampling=PermutationRandomSampling(),
        crossover=OrderCrossover(prob=config.crossover_rate),
        mutation=InversionMutation(prob=config.mutation_rate),
        eliminate_duplicates=False,
    )

    t0 = time.perf_counter()
    res = minimize(
        _P(),
        algorithm,
        ("n_gen", config.generations),
        seed=seed,
        save_history=True,
        verbose=False,
    )
    wall = time.perf_counter() - t0

    history = [float(np.min(h.opt.get("F"))) for h in res.history]
    best = float(np.min(res.F))
    return BenchmarkResult(LIBRARY, name, seed, best, history, wall)


def run_tsp(problem: TSPProblem, config: Config, seed: int) -> BenchmarkResult:
    return _run_permutation(problem.name, problem.n_cities, problem.fn, config, seed)


def run_nqueens(problem: NQueensProblem, config: Config, seed: int) -> BenchmarkResult:
    return _run_permutation(problem.name, problem.n, problem.fn, config, seed)


def run_mo(problem: MOProblem, config: Config, seed: int) -> BenchmarkResult:
    low, high = problem.bounds

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=problem.n_var, n_obj=problem.n_obj, xl=low, xu=high)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = problem.batch_fn(X)

    algorithm = NSGA2(
        pop_size=config.population_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=config.crossover_rate, eta=20),
        mutation=PM(prob=config.mutation_rate, eta=20),
        eliminate_duplicates=True,
    )

    t0 = time.perf_counter()
    res = minimize(
        _P(), algorithm, ("n_gen", config.generations), seed=seed, verbose=False
    )
    wall = time.perf_counter() - t0

    front = [tuple(row) for row in res.F]
    hv = hypervolume(front, problem.ref_point)

    return BenchmarkResult(
        LIBRARY,
        problem.name,
        seed,
        hv,
        [],
        wall,
        minimize=False,
        extra={"pareto_front": front},
    )
