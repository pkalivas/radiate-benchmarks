from __future__ import annotations

import time

import numpy as np
import pygad
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from radiate_benchmarks.adapters.base import BenchmarkResult, Config, hypervolume
from radiate_benchmarks.problems.combinatorial import (
    KnapsackProblem,
    NQueensProblem,
    TSPProblem,
)
from radiate_benchmarks.problems.continuous import ContinuousProblem
from radiate_benchmarks.problems.multiobjective import MOProblem

LIBRARY = "pygad"


def run_continuous(problem: ContinuousProblem, config: Config, seed: int) -> BenchmarkResult:
    low, high = problem.bounds

    def fitness_func(ga, sol, idx):
        return -problem.fn(np.asarray(sol))

    ga = pygad.GA(
        num_generations=config.generations,
        num_parents_mating=config.population_size // 2,
        fitness_func=fitness_func,
        sol_per_pop=config.population_size,
        num_genes=problem.dim,
        init_range_low=low,
        init_range_high=high,
        gene_type=float,
        random_mutation_min_val=low,
        random_mutation_max_val=high,
        mutation_by_replacement=True,
        parent_selection_type="tournament",
        K_tournament=3,
        # TODO(pygad bug): pygad 3.7.0's sbx_crossover only ever computes the
        # "lower" SBX child (never the complementary "upper" one), so every gene
        # drifts monotonically toward the lower bound each generation regardless
        # of fitness -- verified by logging population mean gene value over time.
        # This is why pygad barely moves on ackley/rastrigin/rosenbrock/sphere.
        # Swapping to crossover_type="uniform" fixes it (confirmed ackley: ~20 -> ~3.5)
        # but left as-is for now since this is a pygad-side bug, not our config.
        crossover_type="sbx",
        sbx_crossover_eta=15,
        crossover_probability=config.crossover_rate,
        mutation_type="polynomial",
        polynomial_mutation_eta=20,
        mutation_probability=config.mutation_rate,
        keep_elitism=1,
        save_best_solutions=True,
        random_seed=seed,
        suppress_warnings=True,
    )

    t0 = time.perf_counter()
    ga.run()
    wall = time.perf_counter() - t0

    history = [-f for f in ga.best_solutions_fitness]
    best = -ga.best_solution()[1]
    return BenchmarkResult(LIBRARY, problem.name, seed, best, history, wall)


def run_knapsack(problem: KnapsackProblem, config: Config, seed: int) -> BenchmarkResult:
    def fitness_func(ga, sol, idx):
        return problem.fn(np.asarray(sol, dtype=float))

    ga = pygad.GA(
        num_generations=config.generations,
        num_parents_mating=config.population_size // 2,
        fitness_func=fitness_func,
        sol_per_pop=config.population_size,
        num_genes=problem.n_items,
        gene_space=[0, 1],
        gene_type=int,
        parent_selection_type="tournament",
        K_tournament=3,
        crossover_type="uniform",
        crossover_probability=config.crossover_rate,
        mutation_type="random",
        mutation_probability=config.mutation_rate,
        keep_elitism=1,
        save_best_solutions=True,
        random_seed=seed,
        suppress_warnings=True,
    )

    t0 = time.perf_counter()
    ga.run()
    wall = time.perf_counter() - t0

    history = list(ga.best_solutions_fitness)
    best = ga.best_solution()[1]
    return BenchmarkResult(
        LIBRARY, problem.name, seed, best, history, wall, minimize=False
    )


def _run_permutation(name: str, n: int, fitness_fn, config: Config, seed: int) -> BenchmarkResult:
    def fitness_func(ga, sol, idx):
        return -fitness_fn([int(g) for g in sol])

    ga = pygad.GA(
        num_generations=config.generations,
        num_parents_mating=config.population_size // 2,
        fitness_func=fitness_func,
        sol_per_pop=config.population_size,
        num_genes=n,
        gene_space=list(range(n)),
        gene_type=int,
        allow_duplicate_genes=False,
        parent_selection_type="tournament",
        K_tournament=3,
        crossover_type="uniform",
        crossover_probability=config.crossover_rate,
        mutation_type="random",
        mutation_probability=config.mutation_rate,
        keep_elitism=1,
        save_best_solutions=True,
        random_seed=seed,
        suppress_warnings=True,
    )

    t0 = time.perf_counter()
    ga.run()
    wall = time.perf_counter() - t0

    history = [-f for f in ga.best_solutions_fitness]
    best = -ga.best_solution()[1]
    return BenchmarkResult(LIBRARY, name, seed, best, history, wall)


def run_tsp(problem: TSPProblem, config: Config, seed: int) -> BenchmarkResult:
    return _run_permutation(problem.name, problem.n_cities, problem.fn, config, seed)


def run_nqueens(problem: NQueensProblem, config: Config, seed: int) -> BenchmarkResult:
    return _run_permutation(problem.name, problem.n, problem.fn, config, seed)


def run_mo(problem: MOProblem, config: Config, seed: int) -> BenchmarkResult:
    """PyGAD's built-in NSGA-II support (parent_selection_type='nsga2')."""
    low, high = problem.bounds

    def fitness_func(ga, sol, idx):
        return [-v for v in problem.fn(np.asarray(sol))]

    ga = pygad.GA(
        num_generations=config.generations,
        num_parents_mating=config.population_size // 2,
        fitness_func=fitness_func,
        sol_per_pop=config.population_size,
        num_genes=problem.n_var,
        init_range_low=low,
        init_range_high=high,
        gene_type=float,
        random_mutation_min_val=low,
        random_mutation_max_val=high,
        mutation_by_replacement=True,
        parent_selection_type="nsga2",
        # TODO(pygad bug): same one-sided sbx_crossover issue as run_continuous
        # above -- likely contributor to pygad's weak/zero hypervolume on the MO
        # problems (dtlz2 landed at exactly 0.0 across all trials). Worth
        # revisiting once the crossover bug is addressed.
        crossover_type="sbx",
        sbx_crossover_eta=20,
        crossover_probability=config.crossover_rate,
        mutation_type="polynomial",
        polynomial_mutation_eta=20,
        mutation_probability=config.mutation_rate,
        random_seed=seed,
        suppress_warnings=True,
    )

    t0 = time.perf_counter()
    ga.run()
    wall = time.perf_counter() - t0

    raw = -np.array(ga.last_generation_fitness)
    front_mask = NonDominatedSorting().do(raw, only_non_dominated_front=True)
    front = [tuple(row) for row in raw[front_mask]]
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
