"""Python language runner: every Python library x every problem in the spec x every seed.

Invoked by ``python/run.sh``; reads ``--spec`` and writes ``runs.csv``, ``history.csv``
and ``fronts.csv`` to ``--out`` in the layout described by ``schema.md``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

from adapters import (
    deap_adapter,
    # pygad_adapter,
    pymoo_adapter,
    radiate_adapter,
)
from radiate_benchmarks.problems import from_spec
from radiate_benchmarks.schema import RESULT_FILES, join_list
from radiate_benchmarks.spec import load_spec

LANGUAGE = "python"

LIBRARIES: dict[str, ModuleType] = {
    "radiate": radiate_adapter,
    "deap": deap_adapter,
    "pymoo": pymoo_adapter,
    # "pygad": pygad_adapter,
}

METHOD_BY_KIND = {
    "continuous": "run_continuous",
    "knapsack": "run_knapsack",
    "tsp": "run_tsp",
    "nqueens": "run_nqueens",
    "mo": "run_mo",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = load_spec(args.spec)
    args.out.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[dict]] = {name: [] for name in RESULT_FILES}
    failures = 0

    for problem_spec in spec.problems:
        problem = from_spec(problem_spec)
        method = METHOD_BY_KIND[problem_spec["kind"]]
        for lib_name, module in LIBRARIES.items():
            fn = getattr(module, method, None)
            if fn is None:
                print(f"{problem.name:>12} | {lib_name:>8} | not supported, skipping")
                continue
            version = importlib.metadata.version(lib_name)
            for seed in spec.config.seeds:
                print(f"{problem.name:>12} | {lib_name:>8} | seed={seed}", end=" ", flush=True)
                try:
                    result = fn(problem, spec.config, seed)
                except Exception:  # noqa: BLE001 - keep the benchmark run going
                    failures += 1
                    print("FAILED")
                    traceback.print_exc(file=sys.stdout)
                    continue

                key = {"language": LANGUAGE, "library": lib_name, "problem": problem.name, "seed": seed}
                rows["runs.csv"].append(
                    {
                        **key,
                        "library_version": version,
                        "best_fitness": "" if result.best_fitness is None else repr(float(result.best_fitness)),
                        "best_solution": join_list(result.best_solution),
                        "wall_time_s": repr(result.wall_time_s),
                        "spec_hash": spec.spec_hash,
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }
                )
                rows["history.csv"].extend(
                    {**key, "generation": gen, "best_so_far": repr(float(value))}
                    for gen, value in enumerate(result.history)
                )
                rows["fronts.csv"].extend(
                    {**key, "point": i, "objectives": join_list(point)}
                    for i, point in enumerate(result.front)
                )

                best = "front" if result.best_fitness is None else f"best={result.best_fitness:.4f}"
                print(f"-> {best}  time={result.wall_time_s:.3f}s")

    for filename, columns in RESULT_FILES.items():
        with open(args.out / filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows[filename])

    print(f"wrote {len(rows['runs.csv'])} runs to {args.out}/ ({failures} failed)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
