#!/usr/bin/env python3
"""Derive scalar summaries from the preserved captures; no model execution."""

import collections
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


for folder in sorted(ROOT.iterdir()):
    if not (folder / "outcome.json").exists():
        continue
    samples = read(folder / "os-samples.json")
    events = [json.loads(line) for line in (folder / "diagnostics.jsonl").read_text().splitlines()]
    environment = read(folder / "environment.json")
    baseline_swap = environment["baselineSystemSwapUsedBytes"]
    stages = collections.defaultdict(list)
    for sample in samples:
        stages[sample["stage"]].append(sample)

    def os_summary(rows):
        if not rows:
            return None
        return {"sampleCount": len(rows),
                "firstFootprintBytes": rows[0]["processFootprintBytes"],
                "lastFootprintBytes": rows[-1]["processFootprintBytes"],
                "maxFootprintBytes": max(r["processFootprintBytes"] for r in rows),
                "maxSwapGrowthBytes": max(r["systemSwapUsedBytes"] - baseline_swap for r in rows),
                "pressureCounts": dict(collections.Counter(r["pressureLevel"] for r in rows))}

    requests = []
    for request in read(folder / "requests.json") if (folder / "requests.json").exists() else []:
        observed = [e["fields"] for e in events
                    if e["eventName"] == "requestMemory" and e.get("requestID") == request["requestID"]]
        keys = ("activeMlxBytes", "cachedMlxBytes", "processFootprintBytes",
                "requestFullAttentionLayerCount", "requestFullAttentionLogicalBytes",
                "requestFullAttentionArrayBytes", "requestFullAttentionUnusedArrayBytes",
                "requestCacheRecurrentArrayBytes", "recurrentRewindStateBytes")
        maximum = {k: max(int(e[k]) for e in observed if k in e)
                   for k in keys if any(k in e for e in observed)}
        measured = [e for e in observed if e.get("sampleKind") in
                    ("terminal", "afterRelease", "cancelSignal") or
                    (e.get("sampleKind") in ("phaseBegin", "observation") and
                     e.get("requestCacheMeasuredAtPhase") == e["phase"])]
        requests.append({**request, "os": os_summary(stages[request["label"]]),
                         "eventMaxima": maximum, "measuredFacts": measured,
                         "dflash2Engaged": any(e.get("dflash2Engaged") == "true" for e in observed),
                         "kvBits": sorted({str(e.get("kvBits")) for e in events
                                           if e.get("requestID") == request["requestID"]})})
    allocation = [e["fields"] for e in events if e["eventName"] == "allocationMemory"]
    costs = [s["probeNanoseconds"] for s in samples]
    allocation_costs = [float(e["observationMilliseconds"]) for e in allocation]
    summary = {"outcome": read(folder / "outcome.json"), "os": os_summary(samples),
               "osStages": {name: os_summary(rows) for name, rows in stages.items()},
               "eventCounts": dict(collections.Counter(e["eventName"] for e in events)),
               "requests": requests, "allocationPhases": allocation,
               "probeCosts": {"externalMedianNanoseconds": statistics.median(costs) if costs else None,
                              "externalMaxNanoseconds": max(costs, default=None),
                              "externalTotalNanoseconds": sum(costs),
                              "allocationMedianMilliseconds": statistics.median(allocation_costs)
                              if allocation_costs else None,
                              "allocationMaxMilliseconds": max(allocation_costs, default=None),
                              "allocationTotalMilliseconds": sum(allocation_costs)}}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(folder.name, json.dumps(summary["outcome"]), json.dumps(summary["os"]))
