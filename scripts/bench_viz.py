#!/usr/bin/env python3
"""
Benchmark visualization: reads JSON results from benchmarks/results/ and generates
comparison charts in benchmarks/charts/.

Usage:
    python scripts/bench_viz.py                     # all results
    python scripts/bench_viz.py --last 5            # last 5 runs
    python scripts/bench_viz.py --filter "t=0.6"    # filter by param label
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy required. Install with:")
    print("  pip install matplotlib numpy")
    sys.exit(1)


def load_results(results_dir: Path, last: int = 0, filter_str: str = "") -> list[dict]:
    """Load and optionally filter benchmark result JSON files."""
    files = sorted(results_dir.glob("bench_*.json"), key=lambda f: f.name)
    if filter_str:
        files = [f for f in files if filter_str in f.name]
    if last > 0:
        files = files[-last:]

    results = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            data["_filename"] = f.name
            results.append(data)
    return results


def chart_tool_accuracy(results: list[dict], charts_dir: Path):
    """Chart 1: Tool accuracy by parameter config, grouped by scenario."""
    fig, ax = plt.subplots(figsize=(14, 6))

    configs = [r["metadata"]["sweepLabel"] for r in results]
    scenario_ids = sorted(set(s["id"] for r in results for s in r["scenarios"]))

    x = np.arange(len(configs))
    width = 0.8 / max(len(scenario_ids), 1)

    for i, sid in enumerate(scenario_ids):
        accuracies = []
        for r in results:
            scenario = next((s for s in r["scenarios"] if s["id"] == sid), None)
            accuracies.append(scenario["summary"]["toolAccuracy"] * 100 if scenario else 0)
        offset = (i - len(scenario_ids) / 2 + 0.5) * width
        ax.bar(x + offset, accuracies, width, label=sid)

    ax.set_xlabel("Parameter Config")
    ax.set_ylabel("Tool Accuracy (%)")
    ax.set_title("Tool Accuracy by Parameter Config")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts_dir / "tool_accuracy.png", dpi=150)
    plt.close(fig)
    print("  Generated: tool_accuracy.png")


def chart_duplicate_rate(results: list[dict], charts_dir: Path):
    """Chart 2: Duplicate rate comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    configs = [r["metadata"]["sweepLabel"] for r in results]
    overall_rates = [r["aggregate"]["duplicateRate"] * 100 for r in results]
    s5_rates = []
    for r in results:
        s5 = next((s for s in r["scenarios"] if s["id"] == "S5"), None)
        s5_rates.append(s5["summary"]["duplicateRate"] * 100 if s5 else 0)

    x = np.arange(len(configs))
    width = 0.35
    ax.bar(x - width / 2, overall_rates, width, label="Overall", color="#4a90d9")
    ax.bar(x + width / 2, s5_rates, width, label="S5 (Duplicate Detection)", color="#d94a4a")

    ax.set_xlabel("Parameter Config")
    ax.set_ylabel("Duplicate Rate (%)")
    ax.set_title("Duplicate Tool Call Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts_dir / "duplicate_rate.png", dpi=150)
    plt.close(fig)
    print("  Generated: duplicate_rate.png")


def chart_toks_over_turns(results: list[dict], charts_dir: Path):
    """Chart 3: Tokens/second over turns (for S4 long conversation)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for r in results:
        s4 = next((s for s in r["scenarios"] if s["id"] == "S4"), None)
        if not s4:
            continue
        turns = s4["turns"]
        x = [t["turnIndex"] for t in turns]
        y = [t["performance"]["tokPerSec"] for t in turns]
        label = r["metadata"]["sweepLabel"]
        ax.plot(x, y, marker=".", markersize=3, linewidth=1, label=label)

    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Tokens/Second")
    ax.set_title("Generation Speed Over Turns (S4: Long Conversation)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts_dir / "toks_over_turns.png", dpi=150)
    plt.close(fig)
    print("  Generated: toks_over_turns.png")


def chart_latency_distribution(results: list[dict], charts_dir: Path):
    """Chart 4: Latency distribution (box plot per scenario)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    scenario_ids = sorted(set(s["id"] for r in results for s in r["scenarios"]))
    all_data = []
    labels = []

    for r in results:
        config_label = r["metadata"]["sweepLabel"]
        for sid in scenario_ids:
            scenario = next((s for s in r["scenarios"] if s["id"] == sid), None)
            if not scenario:
                continue
            latencies = [t["performance"]["latencyMs"] for t in scenario["turns"]]
            all_data.append(latencies)
            labels.append(f"{sid}\n{config_label}")

    if all_data:
        bp = ax.boxplot(all_data, labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4a90d9")
            patch.set_alpha(0.6)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Distribution by Scenario")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts_dir / "latency_distribution.png", dpi=150)
    plt.close(fig)
    print("  Generated: latency_distribution.png")


def chart_historical_trend(results: list[dict], charts_dir: Path):
    """Chart 5: Historical trend — tool accuracy and tok/s over time."""
    if len(results) < 2:
        print("  Skipped: historical_trend.png (need >= 2 results)")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    dates = [r["metadata"]["date"][:10] for r in results]
    tool_acc = [r["aggregate"]["overallToolAccuracy"] * 100 for r in results]
    tok_sec = [r["aggregate"]["avgTokPerSec"] for r in results]

    x = range(len(dates))
    ax1.plot(x, tool_acc, "o-", color="#4a90d9", label="Tool Accuracy (%)")
    ax2.plot(x, tok_sec, "s-", color="#d94a4a", label="Avg tok/s")

    ax1.set_xlabel("Run Date")
    ax1.set_ylabel("Tool Accuracy (%)", color="#4a90d9")
    ax2.set_ylabel("Avg tok/s", color="#d94a4a")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.set_title("Historical Benchmark Trend")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts_dir / "historical_trend.png", dpi=150)
    plt.close(fig)
    print("  Generated: historical_trend.png")


def chart_scenario_heatmap(results: list[dict], charts_dir: Path):
    """Chart 6: Scenario pass/fail heatmap."""
    configs = [r["metadata"]["sweepLabel"] for r in results]
    scenario_ids = sorted(set(s["id"] for r in results for s in r["scenarios"]))

    # Build matrix: 0=fail, 0.5=partial, 1.0=pass
    matrix = []
    for r in results:
        row = []
        for sid in scenario_ids:
            scenario = next((s for s in r["scenarios"] if s["id"] == sid), None)
            if not scenario:
                row.append(float("nan"))
            elif scenario["passed"]:
                row.append(1.0)
            else:
                # Partial: some turns passed
                passed_turns = sum(1 for t in scenario["turns"] if t["passed"])
                total_turns = len(scenario["turns"])
                if passed_turns > 0:
                    row.append(0.5)
                else:
                    row.append(0.0)
        matrix.append(row)

    if not matrix:
        print("  Skipped: scenario_heatmap.png (no data)")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(scenario_ids) * 1.2), max(4, len(configs) * 0.5)))

    cmap = mcolors.ListedColormap(["#d94a4a", "#d9d94a", "#4ad94a"])
    bounds = [-0.25, 0.25, 0.75, 1.25]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    data = np.array(matrix)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(scenario_ids)))
    ax.set_xticklabels(scenario_ids)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=8)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Parameter Config")
    ax.set_title("Scenario Pass/Fail Heatmap")

    # Add text annotations
    for i in range(len(configs)):
        for j in range(len(scenario_ids)):
            val = matrix[i][j]
            if val != val:  # NaN
                text = "N/A"
            elif val >= 1.0:
                text = "PASS"
            elif val >= 0.5:
                text = "PARTIAL"
            else:
                text = "FAIL"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(charts_dir / "scenario_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Generated: scenario_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--last", type=int, default=0, help="Only show last N results")
    parser.add_argument("--filter", type=str, default="", help="Filter results by substring in filename")
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory")
    parser.add_argument("--charts-dir", type=str, default=None, help="Charts output directory")
    args = parser.parse_args()

    # Resolve paths relative to project root
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    results_dir = Path(args.results_dir) if args.results_dir else project_dir / "benchmarks" / "results"
    charts_dir = Path(args.charts_dir) if args.charts_dir else project_dir / "benchmarks" / "charts"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    charts_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir, last=args.last, filter_str=args.filter)
    if not results:
        print("No benchmark results found.")
        sys.exit(0)

    print(f"Loaded {len(results)} result(s) from {results_dir}")
    print(f"Generating charts to {charts_dir}...")

    chart_tool_accuracy(results, charts_dir)
    chart_duplicate_rate(results, charts_dir)
    chart_toks_over_turns(results, charts_dir)
    chart_latency_distribution(results, charts_dir)
    chart_historical_trend(results, charts_dir)
    chart_scenario_heatmap(results, charts_dir)

    print("Done!")


if __name__ == "__main__":
    main()
