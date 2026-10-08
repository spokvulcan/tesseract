#!/usr/bin/env python3
"""Summarize DFlash2 ruler reports (scripts/dflash2-ruler.sh --bench-json).

Usage:
  dflash2-ruler-report.py NAME=report.json [NAME=report2.json ...] [--require-identity]

Reports sharing a NAME form one arm; pass them in run order (ABAB). Prints,
per arm, the median of every prefill prompt and decode fixture, the GPU
reference, and in-run identity against the AR reference. With two or more
arms, every later arm is compared with the first: median deltas, whether
each fixture's DFlash2 streams equal the first arm's, and whether its first
departure from AR is no earlier than the first arm's (the baseline's known
bf16 ties).

A change that moves round boundaries (drafting, verification shape) moves
the verify attention's key partitions, so its stream may part from the
first arm's at a bf16 tie. Such an arm passes when its report carries the
teacher-forced AR check (`--bench-check`) and every position where its
stream is not the target's argmax is a tie: a logit gap of at most
TIE_ULPS bf16 ulps (the baseline's own ties against AR are 0-2 ulps).
--require-identity exits nonzero when an arm passes neither way.
"""
import json
import statistics
import sys
from collections import OrderedDict

TIE_ULPS = 2.0


def load(arms_spec):
    arms = OrderedDict()
    for spec in arms_spec:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"expected NAME=report.json, got {spec}")
        with open(path) as f:
            arms.setdefault(name, []).append(json.load(f))
    return arms


def median(values):
    return statistics.median(values) if values else float("nan")


def summarize(reports):
    prefill, fixtures, gpu, ar = OrderedDict(), OrderedDict(), [], OrderedDict()
    identity = OrderedDict()
    streams = OrderedDict()
    forced = OrderedDict()
    for report in reports:
        gpu.extend(r["teraflops"] for r in report["gpuReference"])
        for p in report["prefill"]:
            prefill.setdefault(p["file"], []).append(p)
        for fx in report["fixtures"]:
            fixtures.setdefault(fx["name"], []).extend(fx["runs"])
            if fx.get("ar"):
                ar.setdefault(fx["name"], []).append(fx["ar"])
            if fx.get("forced") is not None:
                forced.setdefault(fx["name"], []).append(fx["forced"]["departures"])
            for run in fx["runs"]:
                if run.get("identity") is not None:
                    identity.setdefault(fx["name"], []).append(run["identity"])
                streams.setdefault(fx["name"], []).append(run["stream"])
    return prefill, fixtures, gpu, ar, identity, streams, forced


def ar_position(label):
    """`MATCH` -> infinity, `DIVERGED at +N` -> N."""
    return float("inf") if label == "MATCH" else int(label.rsplit("+", 1)[1])


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    require_identity = "--require-identity" in sys.argv
    if not args:
        sys.exit(__doc__)
    arms = load(args)
    summaries = OrderedDict((name, summarize(reports)) for name, reports in arms.items())
    base_name = next(iter(summaries))
    failed = False
    for name, (prefill, fixtures, gpu, ar, identity, streams, forced) in summaries.items():
        print(f"== {name}: {len(arms[name])} report(s), gpu reference median "
              f"{median(gpu):.2f} TFLOP/s (min {min(gpu):.2f}, max {max(gpu):.2f})")
        for file, records in prefill.items():
            rates = [r["tokensPerSecond"] for r in records]
            print(f"  prefill {file:16s} {records[0]['promptTokens']:6d} tok  median "
                  f"{median(rates):7.1f} tok/s  runs {', '.join(f'{x:.1f}' for x in rates)}  "
                  f"peak {max(r['peakGB'] for r in records):.1f} GB")
        for fixture, runs in fixtures.items():
            rates = [r["tokensPerSecond"] for r in runs]
            accepted = sorted({(r["accepted"], r["proposed"]) for r in runs})
            line = (f"  decode  {fixture:16s} median {median(rates):6.1f} tok/s  runs "
                    f"{', '.join(f'{x:.1f}' for x in rates)}  "
                    f"tok/round {median([r['tokensPerRound'] for r in runs]):.2f}  "
                    f"ms/round {median([r['msPerRound'] for r in runs]):.1f}  "
                    f"accepted {' '.join(f'{a}/{p}' for a, p in accepted)}")
            if fixture in ar:
                line += f"  ar {median([r['tokensPerSecond'] for r in ar[fixture]]):.1f}"
            if fixture in identity:
                labels = sorted(set(identity[fixture]))
                line += f"  identity {', '.join(labels)}"
            print(line)
            for departures in forced.get(fixture, []):
                print(f"          forced AR: {len(departures)} departure(s)"
                      + "".join(f" +{d['position']} ({d['gapUlps']:.2f} ulp)" for d in departures))
    if len(summaries) > 1:
        base = summaries[base_name]
        for name, summary in list(summaries.items())[1:]:
            print(f"== {name} vs {base_name}")
            for file, records in summary[0].items():
                if file in base[0]:
                    a = median([r["tokensPerSecond"] for r in base[0][file]])
                    b = median([r["tokensPerSecond"] for r in records])
                    print(f"  prefill {file:16s} {a:7.1f} -> {b:7.1f} tok/s  {100 * (b / a - 1):+.1f}%")
            for fixture, runs in summary[1].items():
                if fixture not in base[1]:
                    continue
                a = median([r["tokensPerSecond"] for r in base[1][fixture]])
                b = median([r["tokensPerSecond"] for r in runs])
                ref = base[5][fixture][0]
                diverged = [first_divergence(s, ref) for s in summary[5][fixture]]
                same = all(d is None for d in diverged)
                # A differing stream passes on the forced check alone: every
                # departure from the target's argmax a tie.
                checked = summary[6].get(fixture, [])
                gaps = [d["gapUlps"] for departures in checked for d in departures]
                ties = bool(checked) and all(g <= TIE_ULPS for g in gaps)
                if not same and not ties:
                    failed = failed or require_identity
                # Against AR, a later arm may only part where the first arm
                # parts (a known tie) or later, never earlier.
                ar_note = ""
                if fixture in base[4] and fixture in summary[4]:
                    first = min(ar_position(label) for label in base[4][fixture])
                    later = min(ar_position(label) for label in summary[4][fixture])
                    if later < first:
                        failed = failed or (require_identity and not ties)
                        ar_note = f"  AR identity REGRESSED (+{later} < +{first})"
                    else:
                        ar_note = f"  AR identity held (first part at +{later}, base +{first})"
                if checked:
                    ar_note += (f"  forced AR: {len(gaps)} departure(s), "
                                f"{'all ties' if ties else 'NOT ALL TIES'}"
                                + (f" (max {max(gaps):.2f} ulp)" if gaps else ""))
                print(f"  decode  {fixture:16s} {a:6.1f} -> {b:6.1f} tok/s  {100 * (b / a - 1):+.1f}%  "
                      f"streams {'IDENTICAL' if same else 'DIFFER at ' + str(sorted(set(d for d in diverged if d is not None)))}"
                      + ar_note)
    if failed:
        print("identity check FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
