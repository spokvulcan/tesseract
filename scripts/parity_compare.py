#!/usr/bin/env python3
"""Compare interleaved A/B parity-bench report sets (scripts/parity-ab.sh output).

Checks the experiment-loop quality gate and prints per-context metric deltas:

1. QUALITY GATE (hard fail): generated token IDs must be identical per
   (context, run) between baseline and experiment across ALL rounds.
2. Metrics: prefill tok/s, decode tok/s, peak GB per context; load seconds.
   Deltas are experiment vs baseline in % (positive = better for tok/s,
   negative = better for GB / seconds).

Usage: scripts/parity_compare.py <baseline_dir> <experiment_dir>
Each dir contains round<N>/paro-parity-bench/paro_parity_bench_*.json.
"""

import json
import sys
from pathlib import Path


def load_runs(root: Path) -> tuple:
    """Return ({(context, run): [records per round]}, [load records])."""
    runs = {}
    loads = []
    for report in sorted(root.rglob("paro_parity_bench_*.json")):
        data = json.loads(report.read_text())
        loads.append(data["load"])
        for rec in data["runs"]:
            key = (rec["targetContext"], rec["run"])
            runs.setdefault(key, []).append(rec)
    return runs, loads


def pct(new: float, old: float) -> float:
    return (new - old) / old * 100.0 if old else 0.0


def main() -> int:
    base_dir, exp_dir = Path(sys.argv[1]), Path(sys.argv[2])
    base_runs, base_loads = load_runs(base_dir)
    exp_runs, exp_loads = load_runs(exp_dir)

    if not base_runs or not exp_runs:
        print("FATAL: missing reports on one side", file=sys.stderr)
        return 2

    # 0. Symmetry checks — a hard gate must never pass on partial data
    # (a crashed arm must FAIL the comparison, not silently shrink it).
    base_keys, exp_keys = set(base_runs), set(exp_runs)
    if base_keys != exp_keys:
        print("FATAL: context/run keys differ between arms:", file=sys.stderr)
        for key in sorted(base_keys - exp_keys):
            print(f"  only in baseline:   ctx={key[0]} run={key[1]}", file=sys.stderr)
        for key in sorted(exp_keys - base_keys):
            print(f"  only in experiment: ctx={key[0]} run={key[1]}", file=sys.stderr)
        return 2
    count_mismatch = {
        key: (len(base_runs[key]), len(exp_runs[key]))
        for key in base_keys
        if len(base_runs[key]) != len(exp_runs[key])
    }
    if count_mismatch:
        print("FATAL: per-key round counts differ (crashed arm?):", file=sys.stderr)
        for (ctx, run), (nb, ne) in sorted(count_mismatch.items()):
            print(f"  ctx={ctx} run={run}: baseline {nb} rounds vs experiment {ne}",
                  file=sys.stderr)
        return 2

    # 1. Quality gate — token identity per (context, run), all rounds.
    gate_failures = 0
    compared = 0
    for key in sorted(base_runs):
        for b, e in zip(base_runs[key], exp_runs[key]):
            compared += 1
            if b["tokenIDs"] != e["tokenIDs"]:
                gate_failures += 1
                ctx, run = key
                first_diff = next(
                    (i for i, (x, y) in enumerate(zip(b["tokenIDs"], e["tokenIDs"])) if x != y),
                    min(len(b["tokenIDs"]), len(e["tokenIDs"])),
                )
                print(f"GATE FAIL ctx={ctx} run={run}: token IDs diverge at index {first_diff}")
    print(f"\nQUALITY GATE: {compared} (context,run,round) pairs compared, "
          f"{'PASS — token-identical' if gate_failures == 0 else f'FAIL — {gate_failures} mismatches'}")

    # 2. Metrics per context (mean over rounds).
    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n{'ctx':>6} {'metric':<12} {'baseline':>10} {'experiment':>10} {'delta':>8}")
    for key in sorted(base_runs):
        if key not in exp_runs:
            continue
        ctx, _ = key
        for field, label in [("promptTPS", "prefill t/s"), ("generationTPS", "decode t/s"),
                             ("runPeakGB", "peak GB"), ("promptSeconds", "prompt s"),
                             ("tokenizeSeconds", "tokenize s")]:
            b = mean([r[field] for r in base_runs[key]])
            e = mean([r[field] for r in exp_runs[key]])
            print(f"{ctx:>6} {label:<12} {b:>10.2f} {e:>10.2f} {pct(e, b):>+7.2f}%")

    lb = mean([r["loadSeconds"] for r in base_loads])
    le = mean([r["loadSeconds"] for r in exp_loads])
    if lb and le:
        print(f"{'load':>6} {'seconds':<12} {lb:>10.2f} {le:>10.2f} {pct(le, lb):>+7.2f}%")

    return 1 if gate_failures else 0


if __name__ == "__main__":
    sys.exit(main())
