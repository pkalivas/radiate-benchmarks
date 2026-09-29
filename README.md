# radiate-benchmarks

Head-to-head benchmarks comparing [radiate](https://github.com/pkalivas/radiate) against other
EA/GA libraries, across languages. Each language has its own runner that writes a common CSV
format, and a Python aggregator merges them all into one master dataset for charting.

Currently benchmarked:

| Language | Libraries | Runner needs |
|---|---|---|
| Python | **radiate** (bindings), **DEAP**, **pymoo** (**PyGAD** has an adapter but is disabled, see the SBX bug noted in it) | `uv` |
| Rust | **radiate** (native), built against the local checkout at `../radiate` | `cargo` |
| Java | **Jenetics** (all problems), **jMetal** (multi-objective only) | JDK 25 + Maven (`brew install openjdk maven`); Jenetics 9 requires Java 25 |
| C# | **GeneticSharp** (single-objective only) | .NET 9 SDK |

A language whose toolchain is missing is skipped, not failed.

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
uv run main.py                      # full run: pop=100, gens=200, 10 trials per (library, problem)
uv run main.py --suite continuous   # just one suite
uv run main.py --lang python        # just some languages
uv run main.py --aggregate-only     # no benchmarks: re-merge existing results, redraw charts
uv run main.py --trials 20 --population 200 --generations 300
```

A run has four stages:

1. **Spec**: `main.py` writes `spec/`, the problem instances, shared config and seeds that
   every language reads. Instances are generated once, so every library solves the exact
   same problem.
2. **Languages**: every top-level `<lang>/run.sh` runs in turn (never in parallel, so they
   don't compete for CPU) and writes `results/<lang>/{runs,history,fronts}.csv`. A language
   whose toolchain isn't installed is skipped.
3. **Aggregate**: all languages are merged into `results/master/`. Stale rows from an older
   spec are dropped, every best solution is re-scored against the Python reference fitness
   functions, and hypervolume is computed for every multi-objective front.
   Anything rejected lands in `results/master/rejected.csv` with a reason.
4. **Report**: charts and `summary.csv` / `summary.md` are built from the master CSVs into
   `results/report/`. Every series is a `library (language)` entry, e.g. `radiate (python)`.

`schema.md` is the contract every language runner implements: arguments, exit codes, the
spec format, the CSV columns, and what gets timed.

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
main.py                   orchestrator: spec -> each <lang>/run.sh -> aggregate -> report
schema.md                 the runner contract
radiate_benchmarks/       shared Python, not tied to any one language runner
  problems/               reference problem definitions + fitness functions (+ spec generation)
  spec.py                 writes / loads spec/, computes spec_hash
  schema.py               CSV column layout
  aggregate.py            merge, validate, re-score, hypervolume -> results/master/
  report.py               charts + summary table from results/master/ only
rust/                     Rust language runner (radiate native)
java/                     Java language runner (Jenetics, jMetal; Maven project)
csharp/                   C# language runner (GeneticSharp; .NET 9 project)
python/                   Python language runner
  run.sh
  runner.py               libraries x problems x seeds -> CSVs
  adapters/               one module per library (radiate, DEAP, pymoo, PyGAD)
spec/                     generated each run
results/<lang>/           each language's raw CSVs
results/master/           merged + validated CSVs
results/report/           charts + summary
```

## Adding a library or language

- **Python library**: add `python/adapters/<name>_adapter.py` with `run_continuous`,
  `run_knapsack`, `run_tsp`, `run_nqueens` and/or `run_mo` (leave out any it doesn't
  support), register it in `LIBRARIES` in `python/runner.py`, and add its entry
  (e.g. `"<name> (python)"`) to `ENTRY_ORDER` in `radiate_benchmarks/report.py` to fix its color.
- **New language**: create `<lang>/run.sh` that follows `schema.md`. `main.py` finds it on the
  next run. Add its entries to `ENTRY_ORDER`.
- **New problem**: add it to `radiate_benchmarks/problems/` (spec entry, `from_spec`, reference
  fitness), document its `kind` in `schema.md`, then implement it in each language runner.
