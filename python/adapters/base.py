from __future__ import annotations

from dataclasses import dataclass, field

from radiate_benchmarks.spec import Config

__all__ = ["Config", "RunResult"]


@dataclass
class RunResult:
    library: str
    problem: str
    seed: int
    best_fitness: float | None
    """Final objective value (single-objective); None for multi-objective runs, whose
    score (hypervolume) is computed by the aggregator from ``front``."""
    best_solution: list
    """Genes of the best individual, re-scored by the aggregator. Empty for multi-objective."""
    history: list[float]
    """Best-so-far value per generation, in the same units as ``best_fitness``."""
    wall_time_s: float
    front: list[tuple[float, ...]] = field(default_factory=list)
    """Final Pareto front (objective vectors, all minimized). Multi-objective only."""
