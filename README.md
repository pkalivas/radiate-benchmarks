# radiate-benchmarks

Head-to-head benchmarks comparing [radiate](https://github.com/pkalivas/radiate) against three
industry-standard Python EA/GA libraries: **DEAP**, **pymoo**, and **PyGAD**.

## Suites

| Suite | Problems | What's measured |
|---|---|---|
| `continuous` | Sphere, Rastrigin, Rosenbrock, Ackley (dim=30) | best fitness + convergence curve |
| `combinatorial` | 0/1 Knapsack (50 items), TSP (30 cities), N-Queens (n=20, permutation) | best fitness + convergence curve |
| `multiobjective` | ZDT1, ZDT3, DTLZ2 (3 obj) | final hypervolume (fixed reference point) |

Problem instances (knapsack weights/values, TSP city coordinates) are generated once with a fixed
instance seed, so every library solves the *exact same instance*. The per-trial seed only controls
each EA's own randomness.

## Running

```bash
uv run main.py --quick              # fast smoke test (pop=20, gens=15, 2 trials)
uv run main.py                      # full run: pop=100, gens=150, 10 trials per (library, problem)
uv run main.py --suite continuous   # just one suite
uv run main.py --trials 20 --population 200 --generations 300
```

Results land in `results/`: `summary.csv` / `summary.md` (mean +/- std of best fitness and
wall-clock time per library/problem), `convergence_<problem>.png` for every single-objective
problem, `mo_hypervolume.png` for the multi-objective suite, `speed_comparison.png` (mean
wall-clock time per library, grouped by problem, log-scaled), and
`speed_relative_to_radiate.png` (each other library's time as a multiple of radiate's, per
problem), and `heatmap_overview.png` (problem x library grid, quality and speed side by side,
each cell colored by performance relative to the best library on that row).

## Methodology notes

- **Idiomatic, not identical, configuration.** Each library uses its own recommended operators for
  a given representation (e.g. SBX + polynomial mutation for real-valued problems, order crossover +
  inversion mutation for permutations, NSGA-II for multi-objective) with population size, generation
  count, and crossover/mutation *rates* held equal across libraries. Exact algorithmic parity isn't
  achievable across four different codebases — the goal is "how a competent user would configure
  each library," not literally identical operators.
- **Convergence curves** (`history` on `BenchmarkResult`) are the best-so-far fitness per generation,
  read directly off each library's own per-generation stats/callbacks (DEAP's `Logbook`, pymoo's
  `save_history`, PyGAD's `best_solutions_fitness`, radiate's `on_epoch` subscriber) — no extra
  re-running involved, so the recorded wall-clock time is unaffected.
- **Multi-objective is reported as final hypervolume only**, not a convergence curve. Getting a
  real per-generation Pareto front out of radiate's Python API isn't cheap (it would mean re-running
  from scratch up to each checkpoint generation), and doing that only for radiate would bias its
  measured wall-clock time. Bar charts with error bars across trials are the fairer comparison here.
- Population size should stay a multiple of 4 for the multiobjective suite — DEAP's NSGA-II
  crowded-tournament selection expects it.

## Layout

```
radiate_benchmarks/
  problems/            problem instances + fitness functions, one Python function per problem
  adapters/             one module per library, translating a problem into that library's API
    base.py              BenchmarkResult, Config, hypervolume()
    radiate_adapter.py
    deap_adapter.py
    pymoo_adapter.py
    pygad_adapter.py
  harness.py            wires problems x libraries x seeds together
  report.py             aggregation, plots, summary table
main.py                  CLI entrypoint
```

## Adding a problem or library

- **New problem**: add it to the relevant module in `problems/`, then add a `(problem, method_name)`
  entry to the matching suite in `harness.py`'s `SUITES`, and add a `run_<method_name>` function to
  each adapter.
- **New library**: add `<name>_adapter.py` implementing `run_continuous`, `run_knapsack`, `run_tsp`,
  `run_nqueens`, and `run_mo`, then register it in `harness.py`'s `LIBRARIES` and give it a color slot
  in `report.py`'s `LIBRARY_COLORS`/`LIBRARY_ORDER`.
