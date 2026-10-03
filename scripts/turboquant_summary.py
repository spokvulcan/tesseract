#!/usr/bin/env python3
"""Summarize a `--turboquant-bench` report (issue #603) as Markdown tables.

Usage: scripts/turboquant_summary.py <turboquant_bench_*.json>

Prints quality, reproducibility, decode-speed, memory and prefill tables
across contexts, plus the harness checks. Decode tok/s is the median over the speed rounds, and its
delta is against the unquantized arm of the same context; the rounds run
interleaved (ABBA), so that is the comparison the protocol supports.
"""

import json
import statistics
import sys


def sci(value):
    return f"{value:.2e}" if value else "0"


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def main(path):
    with open(path) as handle:
        report = json.load(handle)
    load = report["load"]
    print(f"Model `{load['modelID']}`, {load['hardware']}, {load['osVersion']}, "
          f"source `{load['sourceRevision']}`, nice {load['nice']}.")
    print(f"Corpus `{report['corpus']['path']}` ({report['corpus']['files']} files, "
          f"sha256 `{report['corpus']['sha256'][:16]}…`), "
          f"{report['options']['maxNew']} decode tokens per pass, "
          f"{report['options']['runs']} speed rounds.\n")

    print("### Quality (teacher-forced against the unquantized cache)\n")
    print("| context | arm | KL mean | KL median | KL p99 | KL max | top-1 agreement | "
          "free-running greedy diverges at |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for context in report["contexts"]:
        arms = list(context["quality"])
        if context.get("noiseFloor"):
            arms.append(context["noiseFloor"])
        for record in arms:
            if record["arm"] == "fp16":
                continue
            divergences = sorted({
                str(s["firstGreedyDivergence"]) if s.get("firstGreedyDivergence") is not None
                else "never"
                for s in context["speed"] if s["scheme"] == record["arm"]
            }) or ["-"]
            mismatches = len([s for s in record["topOneMismatchSteps"] if s > 0])
            print(f"| {context['promptTokens']:,} | {record['arm']} | {sci(record['klMean'])} | "
                  f"{sci(record['klMedian'])} | {sci(record['klP99'])} | {sci(record['klMax'])} | "
                  f"{record['topOneAgreement'] * 100:.1f}% ({mismatches} missed) | "
                  f"{', '.join(divergences)} |")

    print("\n### Reproducibility (free-running greedy streams of the same arm, across rounds)\n")
    print("Every round restores the same snapshot and decodes greedily, so the streams "
          "of one arm should be identical.\n")
    print("| context | arm | rounds | first position where two rounds differ |")
    print("| ---: | --- | ---: | --- |")
    for context in report["contexts"]:
        streams = {}
        for record in context["speed"]:
            streams.setdefault(record["scheme"], []).append(record["tokenIDs"])
        for scheme, runs in streams.items():
            positions = [
                first_difference(runs[i], runs[j])
                for i in range(len(runs)) for j in range(i + 1, len(runs))
            ]
            differing = sorted(p for p in positions if p is not None)
            verdict = ", ".join(str(p) for p in differing) if differing else "identical"
            print(f"| {context['promptTokens']:,} | {scheme} | {len(runs)} | {verdict} |")

    print("\n### Decode speed (production iterator, median of rounds)\n")
    print("| context | arm | decode tok/s | vs unquantized | rounds | switch-over ms | "
          "first-token ms |")
    print("| ---: | --- | ---: | ---: | --- | ---: | ---: |")
    for context in report["contexts"]:
        rates = {}
        for record in context["speed"]:
            rates.setdefault(record["scheme"], []).append(record)
        baseline = statistics.median(r["decodeTokensPerSecond"] for r in rates["fp16"])
        for scheme, records in rates.items():
            rate = statistics.median(r["decodeTokensPerSecond"] for r in records)
            rounds = " / ".join(f"{r['decodeTokensPerSecond']:.2f}" for r in records)
            switch = statistics.median(r.get("switchOverSeconds", 0) for r in records) * 1000
            first = statistics.median(r.get("firstTokenSeconds", 0) for r in records) * 1000
            print(f"| {context['promptTokens']:,} | {scheme} | {rate:.2f} | "
                  f"{(rate / baseline - 1) * 100:+.1f}% | {rounds} | {switch:.0f} | {first:.0f} |")

    print("\n### Memory\n")
    print("| context | arm | KV B/token | vs unquantized | KV allocated GB | prefill peak GB | "
          "decode-phase peak GB | run peak GB |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for context in report["contexts"]:
        baseline = context["quality"][0]["layout"]["liveKVBytesPerToken"]
        for record in context["quality"]:
            layout = record["layout"]
            speeds = [s for s in context["speed"] if s["scheme"] == record["arm"]]
            decode_peak = max(s["decodePhasePeakGB"] for s in speeds)
            run_peak = max(s["runPeakGB"] for s in speeds)
            print(f"| {context['promptTokens']:,} | {record['arm']} | "
                  f"{layout['liveKVBytesPerToken']:,.0f} | "
                  f"{baseline / layout['liveKVBytesPerToken']:.2f}x | "
                  f"{layout['allocatedKVBytes'] / 1e9:.2f} | {context['prefill']['peakGB']:.2f} | "
                  f"{decode_peak:.2f} | {run_peak:.2f} |")

    print("\n### Prefill (shared by every arm)\n")
    print("| context | prompt tokens | seconds | tok/s | peak GB | thermal states seen |")
    print("| ---: | ---: | ---: | ---: | ---: | --- |")
    for context in report["contexts"]:
        prefill = context["prefill"]
        thermal = sorted({s["thermalState"] for s in context["speed"]})
        print(f"| {context['targetTokens']:,} | {context['promptTokens']:,} | "
              f"{prefill['seconds']:.1f} | {prefill['tokensPerSecond']:.1f} | "
              f"{prefill['peakGB']:.2f} | {', '.join(thermal)} |")

    checks = [c for context in report["contexts"] for c in context["checks"]]
    failed = [c for c in checks if not c["passed"]]
    print(f"\nHarness checks: {len(checks) - len(failed)}/{len(checks)} passed.")
    for check in failed:
        print(f"- FAILED: {check['name']} ({check['detail']})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
