#!/usr/bin/env python3
"""Measure retained attention capacity against the next turn's growth (#554, item 6).

Reads the owner's prefix-cache diagnostics (the durable CacheDiagnostics JSONL
sink) and pairs every leaf a request left behind (an ordinary check-in, or a
Leaf Rewind) with the next request that restored from it. Prints one JSON
summary: aggregate scalars and per-observation scalars only. No prompt text,
token ids, request ids or timestamps leave the diagnostics directory.

    python3 compaction_retune.py [--diagnostics DIR] [--repo REPO] > summary.json

The definitions and the decision rule are preregistered in README.md.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import sys

MIB = 1 << 20
THRESHOLD_CAP = 64 * MIB  # AttentionCapacityCompaction.maximumThresholdBytes
SPARE_ROWS_TODAY = 256  # one growth step, what compaction leaves
GROWTH_CAP_ROWS = 4096  # KVCacheGrowth's decode step cap
UUID = re.compile(r"[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}", re.I)


def fields(event):
    out = {}
    for field in event.get("fields", []):
        if isinstance(field, dict):
            out[field["key"]] = field["value"]
        else:
            out[field[0]] = field[1]
    return out


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def benchmark_request_ids(repo):
    ids = set()
    for path in glob.glob(os.path.join(repo, "benchmarks", "**", "*"), recursive=True):
        if not path.endswith((".json", ".md", ".jsonl")) or not os.path.isfile(path):
            continue
        with open(path, errors="ignore") as handle:
            ids.update(match.upper() for match in UUID.findall(handle.read()))
    return ids


def load(diagnostics, excluded):
    files = sorted(glob.glob(os.path.join(diagnostics, "*.jsonl")) +
                   glob.glob(os.path.join(diagnostics, "*.jsonl.old")))
    events, digests = [], {}
    for path in files:
        with open(path, "rb") as handle:
            digests[os.path.basename(path)] = hashlib.sha256(handle.read()).hexdigest()
        with open(path, errors="ignore") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                request = (event.get("requestID") or "").upper()
                if not request or request in excluded:
                    continue
                event["_fields"] = fields(event)
                event["_request"] = request
                events.append(event)
    events.sort(key=lambda e: (e.get("timestamp", 0), e["_fields"].get("sequence", "")))
    # The same event can sit in both the rotated and the current file.
    unique, seen = [], set()
    for event in events:
        key = event.get("id") or (event["_request"], event.get("timestamp"), event.get("eventName"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return unique, digests


def observations(events):
    by_request = {}
    for event in events:
        by_request.setdefault(event["_request"], []).append(event)
    found = []
    for request, stream in by_request.items():
        model = stream[0].get("modelID")
        last_capture = None
        for event in stream:
            name, f = event.get("eventName"), event["_fields"]
            if name == "requestMemory" and f.get("phase") == "capturingLeaf" and \
                    number(f.get("requestFullAttentionLayerCount")):
                last_capture = event
            elif name == "leafStore" and f.get("leafOffset") and f.get("path") in \
                    ("live", "boundary", "direct") and last_capture is not None:
                capture = last_capture["_fields"]
                found.append({
                    "kind": "checkIn", "model": model, "request": request,
                    "time": event.get("timestamp", 0), "offset": int(f["leafOffset"]),
                    "logicalBytes": number(capture.get("requestFullAttentionLogicalBytes")),
                    "unusedBytes": number(capture.get("requestFullAttentionUnusedArrayBytes")),
                    "compactedBytes": number(capture.get("leafCompactedBytes")) or 0.0,
                    "budgetBytes": number(capture.get("treeBudgetBytes")),
                })
                last_capture = None
            elif name == "leafRewind":
                found.append({
                    "kind": "rewind", "model": model, "request": request,
                    "time": event.get("timestamp", 0), "offset": int(f["offset"]),
                    "logicalBytes": number(f.get("fullAttentionLogicalBytes")),
                    "unusedBytes": number(f.get("fullAttentionUnusedArrayBytes")),
                    "compactedBytes": number(f.get("compactedBytes")) or 0.0,
                    "budgetBytes": None,
                })
    return [o for o in found if o["logicalBytes"] and o["offset"] > 0 and o["unusedBytes"] is not None]


def pair(found, events):
    lookups = [e for e in events if e.get("eventName") == "lookup"]
    stores = {}
    for event in events:
        f = event["_fields"]
        if event.get("eventName") == "leafStore" and f.get("path") == "live" and f.get("leafOffset"):
            stores[event["_request"]] = int(f["leafOffset"])
    budgets = {}
    for event in events:
        f = event["_fields"]
        if event.get("eventName") == "requestMemory" and f.get("treeBudgetBytes"):
            budgets.setdefault(event["_request"], number(f["treeBudgetBytes"]))
    for obs in found:
        obs["budgetBytes"] = obs["budgetBytes"] or budgets.get(obs["request"])
        rows = obs["logicalBytes"] / obs["offset"]
        obs["rowBytes"] = rows
        obs["retainedBytes"] = obs["unusedBytes"] + obs["compactedBytes"]
        obs["retainedRows"] = obs["retainedBytes"] / rows
        threshold = min(obs["logicalBytes"] / 4, THRESHOLD_CAP)
        obs["compactable"] = obs["retainedBytes"] > threshold
        nxt = next((e for e in lookups
                    if e.get("timestamp", 0) > obs["time"] and e.get("modelID") == obs["model"]
                    and e["_request"] != obs["request"]
                    and e["_fields"].get("reason") == "hit"
                    and number(e["_fields"].get("snapshotOffset")) == obs["offset"]), None)
        if nxt is None:
            obs["next"] = None
            continue
        f = nxt["_fields"]
        prompt = number(f.get("promptTokens"))
        stored = stores.get(nxt["_request"])
        obs["next"] = {
            "suffix": number(f.get("newTokensToPrefill")),
            "generated": (stored - prompt) if stored is not None and prompt else None,
            "residencySeconds": nxt.get("timestamp", 0) - obs["time"],
        }
    return found


def quantiles(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    def at(q):
        return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]
    return {"n": len(values), "median": statistics.median(values), "p90": at(0.9),
            "max": values[-1], "min": values[0]}


def decide(found):
    eligible = [o for o in found if o["compactable"] and o["next"]]
    if len(eligible) < 5:
        return {"rule": "a", "why": f"{len(eligible)} compactable leaves with a next turn, under 5"}
    wasted = [o for o in eligible
              if SPARE_ROWS_TODAY < o["next"]["suffix"] <= o["retainedRows"]]
    share = len(wasted) / len(eligible)
    if share <= 0.2:
        return {"rule": "a", "why": f"growth copies compaction caused: {share:.2f} <= 0.20"}
    if all(o["kind"] == "checkIn" for o in wasted):
        return {"rule": "b", "why": f"{share:.2f} > 0.20, every one after a check-in"}
    p90 = quantiles([o["next"]["suffix"] for o in eligible])["p90"]
    spare = min(GROWTH_CAP_ROWS, max(SPARE_ROWS_TODAY, -(-int(p90) // 256) * 256))
    return {"rule": "d" if spare == GROWTH_CAP_ROWS else "c", "spareRows": spare,
            "why": f"{share:.2f} > 0.20; spare rows from the next-turn suffix p90 {p90:.0f}"}


def summarize(found, digests, excluded_count):
    paired = [o for o in found if o["next"]]
    report = {
        "diagnosticsFiles": digests, "excludedBenchmarkRequestIDs": excluded_count,
        "observations": len(found), "paired": len(paired),
        "byKind": {}, "decision": decide(found),
        "records": [
            {k: o[k] for k in ("kind", "model", "offset", "rowBytes", "retainedRows",
                                "retainedBytes", "compactedBytes", "compactable", "budgetBytes")}
            | {"next": o["next"]}
            for o in found
        ],
    }
    for kind in ("checkIn", "rewind"):
        group = [o for o in found if o["kind"] == kind]
        pairs = [o for o in group if o["next"]]
        fits = lambda cond: sum(1 for o in pairs if cond(o))
        report["byKind"][kind] = {
            "observations": len(group), "paired": len(pairs),
            "compactable": sum(1 for o in group if o["compactable"]),
            "retainedRows": quantiles([o["retainedRows"] for o in group]),
            "retainedMiB": quantiles([o["retainedBytes"] / MIB for o in group]),
            "retainedShareOfBudget": quantiles(
                [o["retainedBytes"] / o["budgetBytes"] for o in group if o["budgetBytes"]]),
            "nextSuffix": quantiles([o["next"]["suffix"] for o in pairs]),
            "nextGenerated": quantiles([o["next"]["generated"] for o in pairs]),
            "residencySeconds": quantiles([o["next"]["residencySeconds"] for o in pairs]),
            "retainedMiBSeconds": quantiles(
                [o["retainedBytes"] / MIB * o["next"]["residencySeconds"] for o in pairs]),
            "suffixFitsRetained": fits(lambda o: o["next"]["suffix"] <= o["retainedRows"]),
            "suffixFitsTodaysSpare": fits(lambda o: o["next"]["suffix"] <= SPARE_ROWS_TODAY),
            "turnFitsRetained": fits(lambda o: o["next"]["generated"] is not None and
                                     o["next"]["suffix"] + o["next"]["generated"] <= o["retainedRows"]),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", default=os.path.expanduser(
        "~/Library/Application Support/CacheDiagnostics"))
    parser.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    args = parser.parse_args()
    excluded = benchmark_request_ids(args.repo)
    events, digests = load(args.diagnostics, excluded)
    found = pair(observations(events), events)
    json.dump(summarize(found, digests, len(excluded)), sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
