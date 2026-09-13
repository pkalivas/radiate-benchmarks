from __future__ import annotations

import random as pyrandom
import time

import numpy as np
from deap import algorithms, base, creator, tools

from radiate_benchmarks.adapters.base import BenchmarkResult, Config, hypervolume
from radiate_benchmarks.problems.combinatorial import (
    KnapsackProblem,
    NQueensProblem,
    TSPProblem,
)
from radiate_benchmarks.problems.continuous import ContinuousProblem
from radiate_benchmarks.problems.multiobjective import MOProblem

LIBRARY = "deap"


def _ensure_creator(name: str, base_class, **kwargs):
    if not hasattr(creator, name):
        creator.create(name, base_class, **kwargs)
    return getattr(creator, name)


def _bound(low: float, high: float):
    """DEAP's documented decorator idiom for clamping real-valued genes to bounds
    after variation operators that don't respect them (cxBlend, mutGaussian)."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                for i in range(len(child)):
                    child[i] = min(high, max(low, child[i]))
            return offspring

        return wrapper

    return decorator


def run_continuous(
    problem: ContinuousProblem, config: Config, seed: int
) -> BenchmarkResult:
    pyrandom.seed(seed)
    low, high = problem.bounds

    FitnessMin = _ensure_creator("FitnessMin", base.Fitness, weights=(-1.0,))
    Individual = _ensure_creator("IndividualFloat", list, fitness=FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", pyrandom.uniform, low, high)
    toolbox.register(
        "individual", tools.initRepeat, Individual, toolbox.attr_float, problem.dim
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (problem.fn(np.array(ind)),))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register(
        "mutate",
        tools.mutGaussian,
        mu=0.0,
        sigma=0.1 * (high - low),
        indpb=config.mutation_rate,
    )
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.decorate("mate", _bound(low, high))
    toolbox.decorate("mutate", _bound(low, high))

    pop = toolbox.population(n=config.population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    hof = tools.HallOfFame(1)

    t0 = time.perf_counter()
    _, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=config.crossover_rate,
        mutpb=1.0,
        ngen=config.generations,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )
    wall = time.perf_counter() - t0

    # eaSimple is a non-elitist generational GA: the per-generation min can regress
    # (the current best gets lost to selection/mutation), so track a running best via
    # a HallOfFame rather than trusting the last generation's population directly.
    history = list(np.minimum.accumulate(logbook.select("min")))
    return BenchmarkResult(LIBRARY, problem.name, seed, hof[0].fitness.values[0], history, wall)


def run_knapsack(
    problem: KnapsackProblem, config: Config, seed: int
) -> BenchmarkResult:
    pyrandom.seed(seed)

    FitnessMax = _ensure_creator("FitnessMax", base.Fitness, weights=(1.0,))
    Individual = _ensure_creator("IndividualBit", list, fitness=FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", pyrandom.randint, 0, 1)
    toolbox.register(
        "individual", tools.initRepeat, Individual, toolbox.attr_bool, problem.n_items
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (problem.fn(np.array(ind, dtype=float)),))
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=config.mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=config.population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    t0 = time.perf_counter()
    _, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=config.crossover_rate,
        mutpb=1.0,
        ngen=config.generations,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )
    wall = time.perf_counter() - t0

    # See run_continuous: eaSimple has no elitism, so track a running best via
    # HallOfFame instead of trusting the last generation's population max.
    history = list(np.maximum.accumulate(logbook.select("max")))
    return BenchmarkResult(
        LIBRARY, problem.name, seed, hof[0].fitness.values[0], history, wall, minimize=False
    )


def _run_permutation(
    name: str,
    n: int,
    fitness_fn,
    config: Config,
    seed: int,
) -> BenchmarkResult:
    pyrandom.seed(seed)

    FitnessMin = _ensure_creator("FitnessMin", base.Fitness, weights=(-1.0,))
    Individual = _ensure_creator("IndividualPerm", list, fitness=FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", pyrandom.sample, range(n), n)
    toolbox.register("individual", tools.initIterate, Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (fitness_fn(list(ind)),))
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=config.mutation_rate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=config.population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    hof = tools.HallOfFame(1)

    t0 = time.perf_counter()
    _, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=config.crossover_rate,
        mutpb=1.0,
        ngen=config.generations,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )
    wall = time.perf_counter() - t0

    # See run_continuous: eaSimple has no elitism, so track a running best via
    # HallOfFame instead of trusting the last generation's population min.
    history = list(np.minimum.accumulate(logbook.select("min")))
    return BenchmarkResult(LIBRARY, name, seed, hof[0].fitness.values[0], history, wall)


def run_tsp(problem: TSPProblem, config: Config, seed: int) -> BenchmarkResult:
    return _run_permutation(problem.name, problem.n_cities, problem.fn, config, seed)


def run_nqueens(problem: NQueensProblem, config: Config, seed: int) -> BenchmarkResult:
    return _run_permutation(problem.name, problem.n, problem.fn, config, seed)


def run_mo(problem: MOProblem, config: Config, seed: int) -> BenchmarkResult:
    """Canonical DEAP NSGA-II loop (mirrors DEAP's own nsga2 example)."""
    pyrandom.seed(seed)
    low, high = problem.bounds

    fitness_name = f"FitnessMinMO{problem.n_obj}"
    ind_name = f"IndividualMO{problem.n_obj}"
    FitnessMulti = _ensure_creator(
        fitness_name, base.Fitness, weights=tuple([-1.0] * problem.n_obj)
    )
    Individual = _ensure_creator(ind_name, list, fitness=FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", pyrandom.uniform, low, high)
    toolbox.register(
        "individual", tools.initRepeat, Individual, toolbox.attr_float, problem.n_var
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: tuple(problem.fn(np.array(ind))))
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=high, eta=20.0)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=low,
        up=high,
        eta=20.0,
        indpb=config.mutation_rate,
    )
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=config.population_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    pop = toolbox.select(pop, len(pop))

    t0 = time.perf_counter()
    for _ in range(config.generations):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if pyrandom.random() <= config.crossover_rate:
                toolbox.mate(child1, child2)
            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values
            del child2.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop = toolbox.select(pop + offspring, config.population_size)
    wall = time.perf_counter() - t0

    front_inds = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    front = [tuple(ind.fitness.values) for ind in front_inds]
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
