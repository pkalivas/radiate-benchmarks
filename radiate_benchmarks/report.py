"""Summary table and comparison charts, built only from the master CSVs.

Nothing here runs a benchmark: every chart reads ``results/master/*.csv`` (written by
``aggregate.py``), so charts can be iterated on without re-running anything. Each
series is an *entry*, i.e. a ``library (language)`` pair such as ``radiate (python)``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Categorical palette, in its fixed slot order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Fixed entry order: an entry's position here is its palette slot, so its color never
# changes when other entries come and go. New entries get appended here, not generated.
ENTRY_ORDER = [
    "radiate (python)",
    "deap (python)",
    "pymoo (python)",
    "pygad (python)",
    "radiate (rust)",
    "jenetics (java)",
    "jmetal (java)",
    "geneticsharp (csharp)",
]

# The entry every speed-ratio chart is expressed relative to.
BASELINE_ENTRY = "radiate (python)"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"

# Sequential single-hue heat ramp (orange, dark -> light) for the performance heatmap:
# indexed by a 0..1 "relative to best" score, so 0 (worst) is darkest and 1 (best) is lightest.
HEAT_ORANGE = LinearSegmentedColormap.from_list(
    "heat_orange",
    ["#5c1f0a", "#8f3312", "#c24a1c", "#eb6834", "#f19a72", "#f8c9ae", "#fde6d8"],
)



def ordered_entries(entries) -> list[str]:
    """Registered entries in ENTRY_ORDER, then any unregistered ones alphabetically."""
    present = set(entries)
    known = [e for e in ENTRY_ORDER if e in present]
    return known + sorted(present - set(ENTRY_ORDER))


def entry_color(entry: str) -> str:
    if entry in ENTRY_ORDER:
        return PALETTE[ENTRY_ORDER.index(entry)]
    # Unregistered entries borrow the slots after the registered ones, deterministically.
    return PALETTE[(len(ENTRY_ORDER) + sum(map(ord, entry))) % len(PALETTE)]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["problem", "entry", "language", "library"])
        .agg(
            best_mean=("best_fitness", "mean"),
            best_std=("best_fitness", "std"),
            time_mean_s=("wall_time_s", "mean"),
            time_std_s=("wall_time_s", "std"),
            n_trials=("seed", "count"),
        )
        .reset_index()
    )
    grouped["entry"] = pd.Categorical(
        grouped["entry"], categories=ordered_entries(grouped["entry"]), ordered=True
    )
    return grouped.sort_values(["problem", "entry"]).reset_index(drop=True)


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


def plot_convergence(history: pd.DataFrame, problem_name: str, out_path: Path) -> None:
    """Mean best-so-far per generation (+/- 1 std across trials), one line per entry."""
    by_lib: dict[str, list[list[float]]] = {}
    for (entry, _seed), run in history[history["problem"] == problem_name].groupby(["entry", "seed"]):
        by_lib.setdefault(entry, []).append(run.sort_values("generation")["best_so_far"].tolist())

    if not by_lib:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style_axes(ax)

    for lib in ordered_entries(by_lib):
        histories = by_lib[lib]
        min_len = min(len(h) for h in histories)
        arr = np.array([h[:min_len] for h in histories])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(min_len)
        color = entry_color(lib)
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
    mo_df = df[df["suite"] == "multiobjective"]
    if mo_df.empty:
        return

    problems = list(mo_df["problem"].unique())
    libs = ordered_entries(mo_df["entry"])

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style_axes(ax)

    n_libs = len(libs)
    width = 0.8 / n_libs
    x = np.arange(len(problems))

    for i, lib in enumerate(libs):
        means, stds = [], []
        for p in problems:
            vals = mo_df[(mo_df["problem"] == p) & (mo_df["entry"] == lib)]["best_fitness"]
            means.append(vals.mean())
            stds.append(vals.std())
        offset = (i - (n_libs - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width * 0.9,
            yerr=stds,
            color=entry_color(lib),
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


def _format_seconds(t: float) -> str:
    return f"{t * 1000:.1f} ms" if t < 1 else f"{t:.2f} s"


def plot_problem_speed(summary: pd.DataFrame, problem: str, entries: list[str], out_path: Path) -> None:
    """One problem's mean wall-clock time per entry, as horizontal bars (+/- 1 std).

    Rows are sorted fastest first. Every entry in the dataset gets a row; one with no runs
    for this problem (e.g. a library without multi-objective support) goes at the bottom,
    marked with an x at zero instead of a bar. radiate entries have bold labels.
    The axis is linear so bar lengths stay honest; each bar carries its value, which keeps
    the fastest entries readable even when they're 100x shorter than the slowest.
    """
    sub = summary[summary["problem"] == problem].set_index("entry")
    if sub.empty:
        return
    means = sub["time_mean_s"]
    stds = sub["time_std_s"].fillna(0.0)
    x_max = float((means + stds).max())
    pad = 0.015 * x_max
    supported = sorted((e for e in entries if e in sub.index), key=lambda e: float(means.loc[e]))
    entries = supported + [e for e in entries if e not in sub.index]

    fig, ax = plt.subplots(figsize=(7.5, 0.48 * len(entries) + 1.5), facecolor=SURFACE)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)

    for row, entry in enumerate(entries):
        if entry not in sub.index:
            ax.text(pad, row, "✗  not supported", ha="left", va="center", fontsize=9, color=TEXT_SECONDARY)
            continue
        mean, std = float(means.loc[entry]), float(stds.loc[entry])
        ax.barh(
            row,
            mean,
            height=0.62,
            xerr=std,
            color=entry_color(entry),
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1, "capsize": 2},
            zorder=2,
        )
        ax.text(mean + std + pad, row, _format_seconds(mean), ha="left", va="center", fontsize=8.5, color=TEXT_PRIMARY)

    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels(entries)
    for label, entry in zip(ax.get_yticklabels(), entries):
        if entry.startswith("radiate "):
            label.set_fontweight("bold")
            label.set_color(TEXT_PRIMARY)
    ax.set_ylim(len(entries) - 0.35, -0.65)  # first entry on top, room below the last row
    ax.set_xlim(0, x_max * 1.2)
    ax.set_xlabel("mean wall-clock time per run (s)")
    n_trials = int(sub["n_trials"].max())
    ax.set_title(f"Speed: {problem}", color=TEXT_PRIMARY, fontsize=12, loc="left", pad=20)
    ax.text(
        0.0, 1.015, f"mean ± std over {n_trials} seeds, lower is faster",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=TEXT_SECONDARY,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_speedup(summary: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of each other library's mean time as a multiple of radiate's, per problem.

    Radiate is the 1x baseline (dashed reference line) rather than its own bar, since the
    interesting number here is the margin, not radiate's absolute time again.
    """
    if BASELINE_ENTRY not in set(summary["entry"]):
        return

    other_libs = [lib for lib in ordered_entries(summary["entry"]) if lib != BASELINE_ENTRY]
    if not other_libs:
        return

    problems = list(summary["problem"].unique())
    baseline = summary[summary["entry"] == BASELINE_ENTRY].set_index("problem")["time_mean_s"]

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    _style_axes(ax)

    n_libs = len(other_libs)
    width = 0.8 / n_libs
    x = np.arange(len(problems))

    for i, lib in enumerate(other_libs):
        sub = summary[summary["entry"] == lib].set_index("problem")["time_mean_s"]
        ratios = [
            sub.loc[p] / baseline.loc[p] if p in sub.index and p in baseline.index else np.nan
            for p in problems
        ]
        offset = (i - (n_libs - 1) / 2) * width
        ax.bar(x + offset, ratios, width * 0.9, color=entry_color(lib), label=f"{lib} / {BASELINE_ENTRY}")

    ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(problems, rotation=30, ha="right")
    ax.set_ylabel(f"time relative to {BASELINE_ENTRY} (x, log scale)")
    ax.set_title(
        f"Speed relative to {BASELINE_ENTRY} (1x baseline)", color=TEXT_PRIMARY, fontsize=12, loc="left"
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
    if BASELINE_ENTRY not in set(summary["entry"]):
        return

    other_libs = [lib for lib in ordered_entries(summary["entry"]) if lib != BASELINE_ENTRY]
    if not other_libs:
        return

    problems = list(summary["problem"].unique())
    baseline = summary[summary["entry"] == BASELINE_ENTRY].set_index("problem")["time_mean_s"]

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
        sub = summary[summary["entry"] == lib].set_index("problem")["time_mean_s"]
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
            color=entry_color(lib),
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
    ax.set_xlabel(f"speed multiple vs {BASELINE_ENTRY}")

    # Header strip spelling out what each side of the zero line means, so the reader
    # never has to infer direction from the sign of a number.
    ax.text(
        0.0,
        1.04,
        f"◄ {BASELINE_ENTRY} slower",
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
        f"{BASELINE_ENTRY} faster ►",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=TEXT_SECONDARY,
        fontweight="bold",
    )
    ax.set_title(
        f"Speed relative to {BASELINE_ENTRY}",
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
    (1.0 = best = lightest, 0 = worst = darkest, on the same orange heat scale in both panels) so problems with
    wildly different units and "better" directions (minimize vs maximize fitness,
    always-minimize time) become directly comparable at a glance. Cell text shows the
    actual mean value; color carries the relative rank.
    """
    if summary.empty:
        return

    problems = list(summary["problem"].unique())
    libs = ordered_entries(summary["entry"])
    minimize_by_problem = df.groupby("problem")["minimize"].first()
    tick_labels = [lib.replace(" (", "\n(") for lib in libs]

    def build_matrix(value_col: str, force_minimize: bool | None) -> tuple[np.ndarray, np.ndarray]:
        scores = np.full((len(problems), len(libs)), np.nan)
        values = np.full((len(problems), len(libs)), np.nan)
        for i, p in enumerate(problems):
            minimize = force_minimize if force_minimize is not None else bool(minimize_by_problem[p])
            sub = summary[summary["problem"] == p].set_index("entry")[value_col]
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
        im = ax.imshow(scores, cmap=HEAT_ORANGE, vmin=0, vmax=1, aspect="auto")
        # 2px surface gaps between cells so the near-white "best" cells keep a visible edge.
        ax.set_xticks(np.arange(-0.5, len(libs), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(problems), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(len(libs)))
        ax.set_xticklabels(tick_labels, color=TEXT_SECONDARY, fontsize=9)
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
                text_color = SURFACE if scores[i, j] < 0.5 else TEXT_PRIMARY
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
    cbar.set_label("relative to best in row (1.0 = best, lightest)", color=TEXT_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)

    fig.suptitle(
        "Problem x library performance overview", color=TEXT_PRIMARY, fontsize=13, x=0.02, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 0.96, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_report(master_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()

    df = pd.read_csv(master_dir / "runs.csv")
    history = pd.read_csv(master_dir / "history.csv")
    if df.empty:
        print("no runs in the master dataset, nothing to report")
        return

    summary = summary_table(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.md", "w") as f:
        f.write(summary.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    for problem_name in df.loc[df["suite"] != "multiobjective", "problem"].unique():
        plot_convergence(history, problem_name, out_dir / f"convergence_{problem_name}.png")

    plot_mo_bars(df, out_dir / "mo_hypervolume.png")

    entries = ordered_entries(summary["entry"])
    for problem_name in df["problem"].unique():
        plot_problem_speed(summary, problem_name, entries, out_dir / f"speed_{problem_name}.png")
    plot_speedup(summary, out_dir / "speed_relative_to_baseline.png")
    plot_speedup_diverging(summary, out_dir / "speed_relative_to_baseline_linear.png")
    plot_perf_heatmap(df, summary, out_dir / "heatmap_overview.png")

    print(f"wrote report to {out_dir}/")
