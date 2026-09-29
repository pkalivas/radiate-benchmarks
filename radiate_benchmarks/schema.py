"""Column layout of the per-language result CSVs (the contract in schema.md).

Every language runner writes these three files into its ``--out`` directory.
List-valued fields (``best_solution``, ``objectives``) are ``;``-joined numbers,
so no CSV quoting is ever needed.
"""

from __future__ import annotations

RUNS_COLUMNS = [
    "language",
    "library",
    "library_version",
    "problem",
    "seed",
    "best_fitness",
    "best_solution",
    "wall_time_s",
    "spec_hash",
    "timestamp",
]

HISTORY_COLUMNS = [
    "language",
    "library",
    "problem",
    "seed",
    "generation",
    "best_so_far",
]

FRONTS_COLUMNS = [
    "language",
    "library",
    "problem",
    "seed",
    "point",
    "objectives",
]

RESULT_FILES = {
    "runs.csv": RUNS_COLUMNS,
    "history.csv": HISTORY_COLUMNS,
    "fronts.csv": FRONTS_COLUMNS,
}


def _format_number(v) -> str:
    if hasattr(v, "item"):  # numpy scalar -> python scalar
        v = v.item()
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return repr(v)  # shortest round-trippable form, so re-scoring sees the exact value
    return str(v)


def join_list(values) -> str:
    return ";".join(_format_number(v) for v in values)


def split_list(field: str) -> list[float]:
    if not isinstance(field, str) or not field:
        return []
    return [float(v) for v in field.split(";")]
