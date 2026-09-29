"""Orchestrator: spec -> every language's run.sh -> aggregate -> report.

Languages are discovered, not registered: any top-level directory with a ``run.sh``
is a language runner (see schema.md for the contract it must follow).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from radiate_benchmarks.aggregate import MASTER_DIR, aggregate
from radiate_benchmarks.report import write_report
from radiate_benchmarks.spec import SUITES, Config, build_spec, load_spec, write_spec

ROOT = Path(__file__).resolve().parent
EXIT_SKIPPED = 3


def discover_languages() -> list[str]:
    return sorted(p.parent.name for p in ROOT.glob("*/run.sh"))


def run_language(lang: str, spec_dir: Path, results_dir: Path) -> tuple[str, float]:
    """Runs one language's run.sh, streaming its output to the console and to log.txt."""
    out_dir = results_dir / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.csv"):
        stale.unlink()

    cmd = ["bash", str(ROOT / lang / "run.sh"), "--spec", str(spec_dir), "--out", str(out_dir)]
    t0 = time.perf_counter()
    with open(out_dir / "log.txt", "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            print(f"  [{lang}] {line}", end="", flush=True)
        code = proc.wait()
    elapsed = time.perf_counter() - t0

    if code == 0:
        return "ok", elapsed
    if code == EXIT_SKIPPED:
        return "skipped", elapsed
    return f"failed (exit {code})", elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    parser.add_argument("--trials", type=int, default=10, help="random seeds per (library, problem)")
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument(
        "--quick", action="store_true", help="fast smoke-test preset (pop=20, gens=15, 2 trials)"
    )
    parser.add_argument(
        "--lang", nargs="+", metavar="LANG", help="only run these languages (default: all found)"
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="skip benchmarks; re-merge the existing results against the existing spec and redraw charts",
    )
    parser.add_argument("--spec", type=Path, default=ROOT / "spec")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    if args.aggregate_only:
        spec = load_spec(args.spec)
    else:
        population, generations, trials = (
            (20, 15, 2) if args.quick else (args.population, args.generations, args.trials)
        )
        config = Config(
            population_size=population,
            generations=generations,
            seeds=[1010 + i for i in range(trials)],
        )
        spec = build_spec(config, None if args.suite == "all" else [args.suite])
        write_spec(spec, args.spec)
        print(f"spec {spec.spec_hash}: {len(spec.problems)} problems, {config}")

        available = discover_languages()
        languages = args.lang or available
        unknown = sorted(set(languages) - set(available))
        if unknown:
            parser.error(f"no run.sh for {unknown}; found {available}")

        # Sequential on purpose: languages running side by side would compete for CPU
        # and skew each other's wall-clock times.
        statuses = {}
        for lang in languages:
            print(f"\n== {lang} ==")
            statuses[lang] = run_language(lang, args.spec, args.results)

        print()
        for lang, (status, elapsed) in statuses.items():
            mark = {"ok": "✓", "skipped": "–"}.get(status, "✗")
            print(f"{lang:>10}  {mark}  {status}  ({elapsed:.1f}s, see {args.results / lang / 'log.txt'})")

    stats = aggregate(args.results, spec)
    print(
        f"\naggregate: {stats.kept} runs kept from {', '.join(stats.languages)}; "
        f"{stats.stale} stale (other spec) dropped; {stats.rejected} rejected"
    )
    if stats.rejected:
        print(f"  rejected runs and reasons: {args.results / MASTER_DIR / 'rejected.csv'}")

    write_report(args.results / MASTER_DIR, args.results / "report")
    return 1 if stats.rejected else 0


if __name__ == "__main__":
    sys.exit(main())
