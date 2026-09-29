"""The benchmark spec: the single source of truth every language runner reads.

Layout written by ``write_spec``::

    spec/
      config.json            shared EA knobs, seeds, and the spec_hash
      problems/<name>.json   one file per problem instance (see schema.md)

``spec_hash`` covers the config and every problem definition, so a result row
stamped with an old hash can be told apart from one produced against this spec.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from radiate_benchmarks.problems import spec_entries

SUITES = ("continuous", "combinatorial", "multiobjective")


@dataclass
class Config:
    """Shared EA knobs, held constant across libraries for a fair comparison."""

    population_size: int = 100
    generations: int = 200
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    seeds: list[int] = field(default_factory=lambda: [1010 + i for i in range(10)])


@dataclass
class Spec:
    config: Config
    problems: list[dict]
    spec_hash: str

    def problem(self, name: str) -> dict:
        for p in self.problems:
            if p["name"] == name:
                return p
        raise KeyError(name)


def _hash(config: Config, problems: list[dict]) -> str:
    payload = json.dumps({"config": asdict(config), "problems": problems}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def build_spec(config: Config, suites: list[str] | None = None) -> Spec:
    suites = list(suites or SUITES)
    problems = [p for p in spec_entries() if p["suite"] in suites]
    return Spec(config, problems, _hash(config, problems))


def write_spec(spec: Spec, spec_dir: Path) -> None:
    problems_dir = spec_dir / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)
    for stale in problems_dir.glob("*.json"):
        stale.unlink()

    config = {
        **asdict(spec.config),
        "problems": [p["name"] for p in spec.problems],
        "spec_hash": spec.spec_hash,
    }
    (spec_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for p in spec.problems:
        (problems_dir / f"{p['name']}.json").write_text(json.dumps(p, indent=2) + "\n")


def load_spec(spec_dir: Path) -> Spec:
    raw = json.loads((spec_dir / "config.json").read_text())
    names = raw.pop("problems")
    spec_hash = raw.pop("spec_hash")
    config = Config(**raw)
    problems = [
        json.loads((spec_dir / "problems" / f"{name}.json").read_text()) for name in names
    ]
    if _hash(config, problems) != spec_hash:
        raise ValueError(f"{spec_dir} has been edited by hand: spec_hash no longer matches")
    return Spec(config, problems, spec_hash)
