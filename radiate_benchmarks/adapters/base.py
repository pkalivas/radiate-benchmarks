from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pymoo.indicators.hv import HV


def hypervolume(front: list[tuple[float, ...]], ref_point: tuple[float, ...]) -> float:
    """Hypervolume dominated by ``front`` w.r.t. ``ref_point`` (all objectives minimized).

    Points that don't dominate the reference point contribute nothing, so this
    stays well-defined even for the noisy, unconverged fronts seen early in a run.
    """
    if not front:
        return 0.0
    F = np.array(front, dtype=float)
    return float(HV(ref_point=np.array(ref_point, dtype=float))(F))


@dataclass
class Config:
    """Shared EA knobs, held constant across libraries for a fair comparison."""

    population_size: int = 100
    generations: int = 150
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1


@dataclass
class BenchmarkResult:
    library: str
    problem: str
    seed: int
    best_fitness: float
    """Final objective value (single-objective) or final hypervolume (multi-objective)."""
    history: list[float]
    """Best-so-far value per generation, in the same units as ``best_fitness``."""
    wall_time_s: float
    minimize: bool = True
    extra: dict = field(default_factory=dict)
