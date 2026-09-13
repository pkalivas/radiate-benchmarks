"""Aggregate BenchmarkResults into a summary table and comparison plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from radiate_benchmarks.adapters.base import BenchmarkResult

# Fixed categorical color assignment, one slot per library, never cycled/reordered.
LIBRARY_COLORS = {
    "radiate": "#2a78d6",  # blue
    "deap": "#eb6834",  # orange
    "pymoo": "#1baf7a",  # aqua
    "pygad": "#eda100",  # yellow
}
LIBRARY_ORDER = ["radiate", "deap", "pymoo", "pygad"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"

# Sequential single-hue ramp (blue, light -> dark) for magnitude heatmaps.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"],
)

MO_PROBLEM_NAMES = {"zdt1", "zdt3", "dtlz2"}


def to_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "library": r.library,
                "problem": r.problem,
                "seed": r.seed,
                "best_fitness": r.best_fitness,
                "wall_time_s": r.wall_time_s,
                "minimize": r.minimize,
            }
            for r in results
        ]
    )


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["problem", "library"])
        .agg(
            best_mean=("best_fitness", "mean"),
            best_std=("best_fitness", "std"),
            time_mean_s=("wall_time_s", "mean"),
            time_std_s=("wall_time_s", "std"),
            n_trials=("seed", "count"),
        )
        .reset_index()
    )
    grouped["library"] = pd.Categorical(
        grouped["library"], categories=LIBRARY_ORDER, ordered=True
    )
    return grouped.sort_values(["problem", "library"]).reset_index(drop=True)


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def plot_convergence(
    results: list[BenchmarkResult], problem_name: str, out_path: Path
) -> None:
    """Mean best-so-far per generation (+/- 1 std across trials), one line per library."""
    by_lib: dict[str, list[list[float]]] = {}
    for r in results:
        if r.problem == problem_name and r.history:
            by_lib.setdefault(r.library, []).append(r.history)

    if not by_lib:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style_axes(ax)

    for lib in LIBRARY_ORDER:
        if lib not in by_lib:
            continue
        histories = by_lib[lib]
        min_len = min(len(h) for h in histories)
        arr = np.array([h[:min_len] for h in histories])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(min_len)
        color = LIBRARY_COLORS[lib]
        ax.plot(x, mean, color=color, linewidth=2, label=lib)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness so far")
    ax.set_title(f"Convergence: {problem_name}", color=TEXT_PRIMARY, fontsize=12, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_mo_bars(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart of mean final hypervolume (+/- std) per library per MO problem."""
    mo_df = df[df["problem"].isin(MO_PROBLEM_NAMES)]
    if mo_df.empty:
        return

    problems = [p for p in ["zdt1", "zdt3", "dtlz2"] if p in mo_df["problem"].unique()]
    libs = [lib for lib in LIBRARY_ORDER if lib in mo_df["library"].unique()]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style_axes(ax)

    n_libs = len(libs)
    width = 0.8 / n_libs
    x = np.arange(len(problems))

    for i, lib in enumerate(libs):
        means, stds = [], []
        for p in problems:
            vals = mo_df[(mo_df["problem"] == p) & (mo_df["library"] == lib)]["best_fitness"]
            means.append(vals.mean())
            stds.append(vals.std())
        offset = (i - (n_libs - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width * 0.9,
            yerr=stds,
            color=LIBRARY_COLORS[lib],
            label=lib,
            capsize=3,
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(problems)
    ax.set_ylabel("hypervolume (higher is better)")
    ax.set_title("Multi-objective: final hypervolume", color=TEXT_PRIMARY, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_speed_bars(summary: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart of mean wall-clock time (+/- std) per library, one group per problem.

    Log-scaled y-axis since a continuous-problem run and a full MO run can differ by orders
    of magnitude, and both need to stay legible in the same figure.
    """
    if summary.empty:
        return

    problems = list(summary["problem"].unique())
    libs = [lib for lib in LIBRARY_ORDER if lib in summary["library"].unique()]

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    _style_axes(ax)

    n_libs = len(libs)
    width = 0.8 / n_libs
    x = np.arange(len(problems))

    for i, lib in enumerate(libs):
        sub = summary[summary["library"] == lib].set_index("problem")
        means = [sub.loc[p, "time_mean_s"] if p in sub.index else np.nan for p in problems]
        stds = [sub.loc[p, "time_std_s"] if p in sub.index else 0.0 for p in problems]
        offset = (i - (n_libs - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width * 0.9,
            yerr=stds,
            color=LIBRARY_COLORS[lib],
            label=lib,
            capsize=3,
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1},
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(problems, rotation=30, ha="right")
    ax.set_ylabel("wall-clock time (s, log scale)")
    ax.set_title(
        "Speed: mean wall-clock time per run", color=TEXT_PRIMARY, fontsize=12, loc="left"
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_speedup(summary: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of each other library's mean time as a multiple of radiate's, per problem.

    Radiate is the 1x baseline (dashed reference line) rather than its own bar, since the
    interesting number here is the margin, not radiate's absolute time again.
    """
    if "radiate" not in summary["library"].unique():
        return

    other_libs = [
        lib for lib in LIBRARY_ORDER if lib != "radiate" and lib in summary["library"].unique()
    ]
    if not other_libs:
        return

    problems = list(summary["problem"].unique())
    baseline = summary[summary["library"] == "radiate"].set_index("problem")["time_mean_s"]

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    _style_axes(ax)

    n_libs = len(other_libs)
    width = 0.8 / n_libs
    x = np.arange(len(problems))

    for i, lib in enumerate(other_libs):
        sub = summary[summary["library"] == lib].set_index("problem")["time_mean_s"]
        ratios = [
            sub.loc[p] / baseline.loc[p] if p in sub.index and p in baseline.index else np.nan
            for p in problems
        ]
        offset = (i - (n_libs - 1) / 2) * width
        ax.bar(x + offset, ratios, width * 0.9, color=LIBRARY_COLORS[lib], label=f"{lib} / radiate")

    ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(problems, rotation=30, ha="right")
    ax.set_ylabel("time relative to radiate (x, log scale)")
    ax.set_title(
        "Speed relative to radiate (1x baseline)", color=TEXT_PRIMARY, fontsize=12, loc="left"
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_speedup_diverging(summary: pd.DataFrame, out_path: Path) -> None:
    """Horizontal diverging bar chart: how many times faster/slower each library is vs radiate.

    Same underlying ratio as plot_speedup (other's mean time / radiate's mean time), reshaped
    into a signed linear multiple so the reader never has to read a log axis: a ratio >= 1
    (other library slower) becomes +ratio, a ratio < 1 (other library faster) becomes
    -1/ratio. Bars point right for "radiate faster", left for "radiate slower" -- labeled
    directly on the chart -- and each bar is annotated with its own multiple so the reading
    doesn't depend on the axis scale at all.
    """
    if "radiate" not in summary["library"].unique():
        return

    other_libs = [
        lib for lib in LIBRARY_ORDER if lib != "radiate" and lib in summary["library"].unique()
    ]
    if not other_libs:
        return

    problems = list(summary["problem"].unique())
    baseline = summary[summary["library"] == "radiate"].set_index("problem")["time_mean_s"]

    n_libs = len(other_libs)
    n_problems = len(problems)
    fig, ax = plt.subplots(figsize=(8, 0.55 * n_problems * n_libs + 1.5), facecolor=SURFACE)
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)

    bar_height = 0.8 / n_libs
    y = np.arange(n_problems)

    bars = []
    for i, lib in enumerate(other_libs):
        sub = summary[summary["library"] == lib].set_index("problem")["time_mean_s"]
        offset = (i - (n_libs - 1) / 2) * bar_height
        for row, p in enumerate(problems):
            if p not in sub.index or p not in baseline.index:
                continue
            ratio = sub.loc[p] / baseline.loc[p]
            value = ratio if ratio >= 1 else -1.0 / ratio
            bars.append((lib, row, offset, value))

    min_val = min((v for *_, v in bars), default=0.0)
    max_val = max((v for *_, v in bars), default=0.0)
    pad = 0.18 * max(abs(min_val), abs(max_val), 1.0)

    seen_libs = set()
    for lib, row, offset, value in bars:
        ypos = row + offset
        ax.barh(
            ypos,
            value,
            bar_height * 0.9,
            color=LIBRARY_COLORS[lib],
            label=lib if lib not in seen_libs else None,
        )
        seen_libs.add(lib)
        label_x = value + (pad * 0.1 if value >= 0 else -pad * 0.1)
        ax.text(
            label_x,
            ypos,
            f"{abs(value):.1f}x",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )

    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1)
    ax.set_xlim(min(min_val, 0.0) - pad, max(max_val, 0.0) + pad)
    ax.set_yticks(y)
    ax.set_yticklabels(problems)
    ax.invert_yaxis()
    ax.set_xlabel("speed multiple vs radiate")

    # Header strip spelling out what each side of the zero line means, so the reader
    # never has to infer direction from the sign of a number.
    ax.text(
        0.0,
        1.04,
        "◄ radiate slower",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=TEXT_SECONDARY,
        fontweight="bold",
    )
    ax.text(
        1.0,
        1.04,
        "radiate faster ►",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=TEXT_SECONDARY,
        fontweight="bold",
    )
    ax.set_title(
        "Speed relative to radiate",
        color=TEXT_PRIMARY,
        fontsize=12,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _relative_score(value: float, best_value: float, minimize: bool) -> float:
    """1.0 = matches the best value on this row, lower = further behind it.

    Handles a best_value of exactly 0 (e.g. n-queens fully solved) separately, since
    the usual ratio would divide by zero there.
    """
    if best_value == 0:
        return 1.0 if value == 0 else 0.0
    return (best_value / value) if minimize else (value / best_value)


def plot_perf_heatmap(df: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    """Problem x library heatmap: quality and speed side by side.

    Each cell is colored by its performance relative to the best library on that row
    (1.0 = best, on the same sequential-blue scale in both panels) so problems with
    wildly different units and "better" directions (minimize vs maximize fitness,
    always-minimize time) become directly comparable at a glance. Cell text shows the
    actual mean value; color carries the relative rank.
    """
    if summary.empty:
        return

    problems = list(summary["problem"].unique())
    libs = [lib for lib in LIBRARY_ORDER if lib in summary["library"].unique()]
    minimize_by_problem = df.groupby("problem")["minimize"].first()

    def build_matrix(value_col: str, force_minimize: bool | None) -> tuple[np.ndarray, np.ndarray]:
        scores = np.full((len(problems), len(libs)), np.nan)
        values = np.full((len(problems), len(libs)), np.nan)
        for i, p in enumerate(problems):
            minimize = force_minimize if force_minimize is not None else bool(minimize_by_problem[p])
            sub = summary[summary["problem"] == p].set_index("library")[value_col]
            if sub.empty:
                continue
            best_value = sub.min() if minimize else sub.max()
            for j, lib in enumerate(libs):
                if lib not in sub.index:
                    continue
                val = sub.loc[lib]
                scores[i, j] = _relative_score(val, best_value, minimize)
                values[i, j] = val
        return scores, values

    quality_scores, quality_values = build_matrix("best_mean", force_minimize=None)
    speed_scores, speed_values = build_matrix("time_mean_s", force_minimize=True)

    fig, (ax_quality, ax_speed, cax) = plt.subplots(
        1,
        3,
        figsize=(12, 0.45 * len(problems) + 2.2),
        facecolor=SURFACE,
        gridspec_kw={"width_ratios": [10, 10, 0.5], "wspace": 0.35},
    )

    panels = [
        (ax_quality, quality_scores, quality_values, "Quality (best fitness, relative to best)", "{:.3g}"),
        (ax_speed, speed_scores, speed_values, "Speed (mean time, relative to fastest)", "{:.3f}s"),
    ]
    im = None
    for ax, scores, values, title, fmt in panels:
        im = ax.imshow(scores, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(libs)))
        ax.set_xticklabels(libs, color=TEXT_SECONDARY, fontsize=9)
        ax.set_yticks(range(len(problems)))
        ax.set_yticklabels(problems, color=TEXT_SECONDARY, fontsize=9)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(problems)):
            for j in range(len(libs)):
                if np.isnan(scores[i, j]):
                    continue
                text_color = SURFACE if scores[i, j] >= 0.6 else TEXT_PRIMARY
                ax.text(
                    j,
                    i,
                    fmt.format(values[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("relative to best in row (1.0 = best)", color=TEXT_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)

    fig.suptitle(
        "Problem x library performance overview", color=TEXT_PRIMARY, fontsize=13, x=0.02, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 0.96, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_report(results: list[BenchmarkResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = to_dataframe(results)

    summary = summary_table(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.md", "w") as f:
        f.write(summary.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    for problem_name in df["problem"].unique():
        if problem_name in MO_PROBLEM_NAMES:
            continue
        plot_convergence(results, problem_name, out_dir / f"convergence_{problem_name}.png")

    plot_mo_bars(df, out_dir / "mo_hypervolume.png")

    plot_speed_bars(summary, out_dir / "speed_comparison.png")
    plot_speedup(summary, out_dir / "speed_relative_to_radiate.png")
    plot_speedup_diverging(summary, out_dir / "speed_relative_to_radiate_linear.png")
    plot_perf_heatmap(df, summary, out_dir / "heatmap_overview.png")

    print(f"Wrote report to {out_dir}/")
