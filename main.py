import argparse
from pathlib import Path

from radiate_benchmarks.adapters.base import Config
from radiate_benchmarks.harness import SUITES, run_suite
from radiate_benchmarks.report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark radiate against DEAP, pymoo, and PyGAD."
    )
    parser.add_argument("--suite", choices=[*SUITES.keys(), "all"], default="all")
    parser.add_argument(
        "--trials", type=int, default=10, help="number of random seeds per (library, problem)"
    )
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="fast smoke-test preset (small population/generations/trials)",
    )
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.quick:
        population, generations, trials = 20, 15, 2
    else:
        population, generations, trials = args.population, args.generations, args.trials

    config = Config(population_size=population, generations=generations)
    seeds = list(range(trials))

    suite_names = list(SUITES.keys()) if args.suite == "all" else [args.suite]
    results = []
    for suite_name in suite_names:
        results.extend(run_suite(suite_name, config, seeds))

    write_report(results, args.out)


if __name__ == "__main__":
    main()
