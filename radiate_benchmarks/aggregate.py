"""Merge every language's result CSVs into one validated master dataset.

Reads ``results/<language>/{runs,history,fronts}.csv`` and writes ``results/master/``:

- rows stamped with a different ``spec_hash`` (stale runs) are dropped;
- single-objective runs are re-scored: ``best_solution`` is evaluated with the
  reference fitness function and must reproduce the reported ``best_fitness``,
  which catches a fitness function ported incorrectly to another language;
- multi-objective runs get their ``best_fitness`` here, as the hypervolume of
  their final front against the spec's reference point, computed identically
  for every library.

Rejected rows go to ``master/rejected.csv`` with a reason instead of vanishing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

from radiate_benchmarks.problems import reference_score
from radiate_benchmarks.schema import FRONTS_COLUMNS, HISTORY_COLUMNS, RUNS_COLUMNS, split_list
from radiate_benchmarks.spec import Spec

MASTER_DIR = "master"
RUN_KEY = ["language", "library", "problem", "seed"]

# Relative tolerance for re-scoring. Loose enough for float32 genes or a different
# summation order in another language; tight enough to catch a wrong formula.
RESCORE_RTOL = 1e-6
RESCORE_ATOL = 1e-6


@dataclass
class AggregateStats:
    kept: int
    stale: int
    rejected: int
    languages: list[str]


def hypervolume(front: list[list[float]], ref_point: list[float]) -> float:
    """Hypervolume dominated by ``front`` w.r.t. ``ref_point`` (all objectives minimized).

    Points that don't dominate the reference point contribute nothing, so this
    stays well-defined even for the noisy, unconverged fronts seen early in a run.
    """
    if not front:
        return 0.0
    return float(HV(ref_point=np.array(ref_point, dtype=float))(np.array(front, dtype=float)))


def entry_label(library: str, language: str) -> str:
    """How a (library, language) pair is named in every table and chart."""
    return f"{library} ({language})"


def _language_dirs(results_dir: Path) -> list[Path]:
    return sorted(
        d
        for d in results_dir.iterdir()
        if d.is_dir() and d.name != MASTER_DIR and (d / "runs.csv").exists()
    )


# Read as text no matter what they look like: a spec_hash of all digits, or a
# library_version of "1.10", would otherwise be parsed as a number and corrupted.
TEXT_COLUMNS = [
    "language",
    "library",
    "library_version",
    "problem",
    "best_solution",
    "spec_hash",
    "timestamp",
    "objectives",
]


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, dtype={c: "string" for c in TEXT_COLUMNS if c in columns})
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)} (see schema.md)")
    return df[columns]


def _validate(row: pd.Series, spec: Spec, fronts: dict[tuple, list[list[float]]]) -> tuple[float | None, str | None]:
    """Returns (score, None) for a valid run or (None, reason) for a rejected one."""
    problem = spec.problem(row["problem"])

    if problem["kind"] == "mo":
        front = fronts.get(tuple(row[k] for k in RUN_KEY), [])
        if not front:
            return None, "multi-objective run has no front in fronts.csv"
        if any(len(p) != problem["n_obj"] for p in front):
            return None, f"front points must have {problem['n_obj']} objectives"
        return hypervolume(front, problem["ref_point"]), None

    reported = row["best_fitness"]
    if pd.isna(reported):
        return None, "best_fitness is empty"
    solution = split_list(row["best_solution"])
    if not solution:
        return None, "best_solution is empty, so it can't be re-scored"
    try:
        rescored = reference_score(problem, solution)
    except Exception as exc:  # noqa: BLE001 - a malformed solution is a rejection, not a crash
        return None, f"re-scoring failed: {exc}"
    if not math.isclose(rescored, reported, rel_tol=RESCORE_RTOL, abs_tol=RESCORE_ATOL):
        return None, f"re-score mismatch: reported {reported!r}, reference fitness gives {rescored!r}"
    return float(reported), None


def aggregate(results_dir: Path, spec: Spec) -> AggregateStats:
    lang_dirs = _language_dirs(results_dir)
    if not lang_dirs:
        raise FileNotFoundError(f"no results/<language>/runs.csv found under {results_dir}")
    runs = pd.concat(
        [_read(d / "runs.csv", RUNS_COLUMNS) for d in lang_dirs],
        ignore_index=True,
    )
    history = pd.concat(
        [_read(d / "history.csv", HISTORY_COLUMNS) for d in lang_dirs],
        ignore_index=True,
    )
    fronts_df = pd.concat(
        [_read(d / "fronts.csv", FRONTS_COLUMNS) for d in lang_dirs],
        ignore_index=True,
    )

    known_problems = {p["name"] for p in spec.problems}
    fresh = (runs["spec_hash"] == spec.spec_hash) & runs["problem"].isin(known_problems)
    stale = int((~fresh).sum())
    runs = runs[fresh].copy()

    fronts: dict[tuple, list[list[float]]] = {}
    for row in fronts_df.sort_values("point").itertuples(index=False):
        key = (row.language, row.library, row.problem, row.seed)
        fronts.setdefault(key, []).append(split_list(row.objectives))

    scores, reasons = [], []
    for _, row in runs.iterrows():
        score, reason = _validate(row, spec, fronts)
        scores.append(score)
        reasons.append(reason)
    runs["best_fitness"] = scores
    runs["reason"] = reasons

    duplicated = runs.duplicated(RUN_KEY, keep=False) & runs["reason"].isna()
    runs.loc[duplicated, "reason"] = "duplicate (language, library, problem, seed)"

    rejected = runs[runs["reason"].notna()]
    kept = runs[runs["reason"].isna()].drop(columns="reason")

    problem_meta = pd.DataFrame(
        [{"problem": p["name"], "suite": p["suite"], "minimize": p["minimize"]} for p in spec.problems]
    )
    kept = kept.merge(problem_meta, on="problem", how="left")
    kept.insert(2, "entry", [entry_label(lib, lang) for lib, lang in zip(kept["library"], kept["language"])])
    history = history.merge(kept[[*RUN_KEY, "entry"]], on=RUN_KEY, how="inner")
    fronts_df = fronts_df.merge(kept[[*RUN_KEY, "entry"]], on=RUN_KEY, how="inner")

    out = results_dir / MASTER_DIR
    out.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out / "runs.csv", index=False)
    history.to_csv(out / "history.csv", index=False)
    fronts_df.to_csv(out / "fronts.csv", index=False)
    rejected[[*RUN_KEY, "reason"]].to_csv(out / "rejected.csv", index=False)

    return AggregateStats(
        kept=len(kept),
        stale=stale,
        rejected=len(rejected),
        languages=[d.name for d in lang_dirs],
    )
