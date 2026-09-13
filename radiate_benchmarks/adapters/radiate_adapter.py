from __future__ import annotations

import time

import numpy as np
import radiate as rd

from radiate_benchmarks.adapters.base import BenchmarkResult, Config, hypervolume
from radiate_benchmarks.problems.combinatorial import (
    KnapsackProblem,
    NQueensProblem,
    TSPProblem,
)
from radiate_benchmarks.problems.continuous import ContinuousProblem
from radiate_benchmarks.problems.multiobjective import MOProblem

LIBRARY = "radiate"


def run_continuous(
    problem: ContinuousProblem, config: Config, seed: int
) -> BenchmarkResult:
    rd.random.seed(seed)
    history: list[float] = []

    @rd.on_epoch
    def collect(event: rd.EngineEvent) -> None:
        score = event.score()
        if isinstance(score, (list, tuple, np.ndarray)):
            history.append(score[0])

    engine = (
        rd.Engine.float(problem.dim, init_range=problem.bounds, use_numpy=True)
        .fitness(problem.fn)
        .minimizing()
        .subscribe(collect)
        .size(config.population_size)
        .select(rd.Select.tournament(k=3), rd.Select.elite())
        .alter(
            rd.Cross.uniform(config.crossover_rate),
            rd.Mutate.arithmetic(config.mutation_rate),
        )
        .limit(rd.Limit.generations(config.generations))
    )

    t0 = time.perf_counter()
    result = engine.run()
    wall = time.perf_counter() - t0

    return BenchmarkResult(
        LIBRARY, problem.name, seed, result.score()[0], history, wall
    )


def run_knapsack(
    problem: KnapsackProblem, config: Config, seed: int
) -> BenchmarkResult:
    rd.random.seed(seed)
    history: list[float] = []

    @rd.on_epoch
    def collect(event: rd.EngineEvent) -> None:
        score = event.score()
        if isinstance(score, (list, tuple, np.ndarray)):
            history.append(score[0])

    def fit(bits) -> float:
        return problem.fn(np.asarray(bits, dtype=float))

    engine = (
        rd.Engine.bit(problem.n_items, use_numpy=True)
        .fitness(fit)
        .maximizing()
        .subscribe(collect)
        .size(config.population_size)
        .select(rd.Select.tournament(k=3), rd.Select.elite())
        .alter(
            rd.Cross.uniform(config.crossover_rate),
            rd.Mutate.uniform(config.mutation_rate),
        )
        .limit(rd.Limit.generations(config.generations))
    )

    t0 = time.perf_counter()
    result = engine.run()
    wall = time.perf_counter() - t0

    return BenchmarkResult(
        LIBRARY, problem.name, seed, result.score()[0], history, wall, minimize=False
    )


def run_tsp(problem: TSPProblem, config: Config, seed: int) -> BenchmarkResult:
    rd.random.seed(seed)
    history: list[float] = []

    @rd.on_epoch
    def collect(event: rd.EngineEvent) -> None:
        score = event.score()
        if isinstance(score, (list, tuple, np.ndarray)):
            history.append(score[0])

    engine = (
        rd.Engine.permutation(list(range(problem.n_cities)))
        .fitness(problem.fn)
        .minimizing()
        .subscribe(collect)
        .size(config.population_size)
        .select(rd.Select.tournament(k=3), rd.Select.elite())
        .alter(
            rd.Cross.pmx(config.crossover_rate),
            rd.Mutate.inversion(config.mutation_rate),
        )
        .limit(rd.Limit.generations(config.generations))
    )

    t0 = time.perf_counter()
    result = engine.run()
    wall = time.perf_counter() - t0

    return BenchmarkResult(
        LIBRARY, problem.name, seed, result.score()[0], history, wall
    )


def run_nqueens(problem: NQueensProblem, config: Config, seed: int) -> BenchmarkResult:
    rd.random.seed(seed)
    history: list[float] = []

    @rd.on_epoch
    def collect(event: rd.EngineEvent) -> None:
        score = event.score()
        if isinstance(score, (list, tuple, np.ndarray)):
            history.append(score[0])

    engine = (
        rd.Engine.permutation(list(range(problem.n)))
        .fitness(problem.fn)
        .minimizing()
        .subscribe(collect)
        .size(config.population_size)
        .select(rd.Select.tournament(k=3), rd.Select.elite())
        .alter(
            rd.Cross.pmx(config.crossover_rate),
            rd.Mutate.inversion(config.mutation_rate),
        )
        .limit(rd.Limit.generations(config.generations))
    )

    t0 = time.perf_counter()
    result = engine.run()
    wall = time.perf_counter() - t0

    return BenchmarkResult(
        LIBRARY, problem.name, seed, result.score()[0], history, wall
    )


def run_mo(problem: MOProblem, config: Config, seed: int) -> BenchmarkResult:
    rd.random.seed(seed)

    def fit(x) -> list[float]:
        return list(problem.fn(np.asarray(x)))

    engine = (
        rd.Engine.float(problem.n_var, init_range=problem.bounds, use_numpy=True)
        .fitness(fit)
        .objective(
            *([rd.MIN] * problem.n_obj),
            front_range=(config.population_size, config.population_size + 50),
        )
        .size(config.population_size)
        .select(rd.Select.tournament(k=3), rd.Select.nsga2())
        .alter(
            rd.Cross.sbx(config.crossover_rate, 2.0),
            rd.Mutate.uniform(config.mutation_rate),
        )
        .limit(rd.Limit.generations(config.generations))
    )

    t0 = time.perf_counter()
    result = engine.run()
    wall = time.perf_counter() - t0

    front = [tuple(member.score()) for member in result.front()]
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
