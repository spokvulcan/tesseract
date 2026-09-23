#!/usr/bin/env python3
"""Summarize the #554 gate's campaigns into committed scalars.

    python3 summarize.py RESULTS_DIR > summary.json

Reads each campaign directory the gate wrote (the profile's and the probe's
records, OS samples and outcomes) and the durable CacheDiagnostics sink for
the reserve's lane counts. Prints per-request scalars and the pass-rule
comparison. No prompt text, token ids, request ids or timestamps are
printed.
"""
import datetime
import glob
import json
import os
import pathlib
import sys

SINK = pathlib.Path.home() / "Library/Application Support/CacheDiagnostics"
MODEL = "qwen3.8-27b"


def load(path, default=None):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return default


def number(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def fields(event):
    raw = event.get("fields", {})
    if isinstance(raw, dict):
        return raw
    return {item["key"]: item["value"] for item in raw}


def seconds(timestamp):
    return datetime.datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).timestamp()


def restore_facts(events):
    lookup = next((fields(e) for e in events if e["eventName"] == "lookup"), {})
    stores = [fields(e) for e in events if e["eventName"] == "leafStore"]
    rewinds = [fields(e) for e in events if e["eventName"] == "leafRewind"]
    keep = ("reason", "snapshotOffset", "hydratedFromSSD", "restoreMode", "copyReason",
            "copyRefusal", "newTokensToPrefill")
    return {
        "lookup": {k: lookup[k] for k in keep if k in lookup},
        "stores": [{k: s[k] for k in ("path", "source", "restoreMode", "copyReason", "copyRefusal",
                                      "leafOffset", "compactedBytes") if k in s} for s in stores],
        "rewinds": [{k: number(r.get(k)) for k in ("offset", "compactedBytes",
                                                    "fullAttentionArrayBytes",
                                                    "fullAttentionLogicalBytes")} for r in rewinds],
    }


def memory_facts(samples):
    def peak(key):
        values = [number(s.get(key)) for s in samples]
        return max((v for v in values if v is not None), default=None)
    settled = next((s for s in samples if s.get("sampleKind") == "afterRelease"), {})
    terminal = next((s for s in samples if s.get("sampleKind") == "terminal"), {})
    return {
        "peakActiveMlxBytes": peak("sampledRequestPeakActiveMlxBytes") or peak("activeMlxBytes"),
        "peakFootprintBytes": peak("sampledRequestPeakFootprintBytes") or peak("processFootprintBytes"),
        "settledActiveMlxBytes": number(settled.get("activeMlxBytes")),
        "settledSampleFootprintBytes": number(settled.get("processFootprintBytes")),
        "terminalTreeLeaseCount": number(terminal.get("treeLeaseCount")),
        "settledTreeLeaseCount": number(settled.get("treeLeaseCount")),
        "releasingRequestPhases": sum(1 for s in samples if s.get("phase") == "releasingRequest"
                                      and s.get("sampleKind") == "phaseBegin"),
    }


def os_peak(os_samples, label):
    values = [s["processFootprintBytes"] for s in os_samples
              if str(s.get("stage", "")).split(":")[0] == label]
    return max(values, default=None)


def lanes(start, end):
    """The reserve's lane counts the durable sink measured in a window."""
    counts = {}
    for path in sorted(glob.glob(str(SINK / "*.jsonl")) + glob.glob(str(SINK / "*.jsonl.old"))):
        with open(path, errors="ignore") as handle:
            for line in handle:
                if '"budgetMeasure"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("modelID") not in (None, MODEL):
                    continue
                if not start <= seconds(event.get("timestamp")) <= end:
                    continue
                value = fields(event).get("lanes")
                counts[value] = counts.get(value, 0) + 1
    return counts


def campaign(directory):
    name = directory.name
    run, arm, kind = name.split("-")[:3]
    outcome = load(directory / "outcome.json", {})
    environment = load(directory / "environment.json", {})
    os_samples = load(directory / "os-samples.json", [])
    records = load(directory / "requests.json", [])
    events = load(directory / "events.json", [])
    if not events and (directory / "diagnostics.jsonl").exists():
        # The probe keeps the sink's own lines.
        with open(directory / "diagnostics.jsonl", errors="ignore") as handle:
            for line in handle:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
    by_request = {}
    for event in events:
        by_request.setdefault(event.get("requestID"), []).append(event)
    requests = []
    for record in records:
        related = record.get("events") or by_request.get(record.get("requestID"), [])
        samples = [fields(e) for e in related if e["eventName"] == "requestMemory"
                   and e.get("requestID") == record.get("requestID")]
        usage = record.get("usage") or {}
        requests.append({
            "label": record["label"],
            "promptTokens": usage.get("prompt_tokens"),
            "cachedTokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "generatedTokens": usage.get("completion_tokens", record.get("generatedDeltas")),
            "cancelled": record.get("cancelled"),
            "responseSeconds": round(record.get("responseSeconds") or 0, 2),
            **restore_facts(related),
            **memory_facts(samples),
            "osPeakFootprintBytes": os_peak(os_samples, record["label"]),
        })
        # The profile samples the settled footprint itself; the probe relies
        # on the request's after-release sample.
        requests[-1]["settledFootprintBytes"] = (
            record.get("settledFootprintBytes") or requests[-1]["settledSampleFootprintBytes"])
    window = None
    if environment.get("startUTC"):
        start = seconds(environment["startUTC"])
        end = (seconds(outcome["endUTC"]) if outcome.get("endUTC")
               else start + (outcome.get("elapsedSeconds") or 0) + 5)
        window = lanes(start, end)
    keep = ("failure", "resourceStops", "exitCode", "appExitCode", "completedRequests",
            "elapsedSeconds", "maxSampledFootprintBytes", "minSampledAvailableBytes",
            "maxSystemSwapGrowthBytes", "maxPressureLevel", "diagnosticsRotations")
    return name, {
        "run": run, "arm": arm, "kind": kind,
        "outcome": {k: outcome[k] for k in keep if k in outcome},
        "initialAvailableBytes": environment.get("initialAvailableBytes"),
        "requests": requests,
        "reserveLanes": window,
    }


def compare(campaigns):
    """The pass rule's memory half, per workload step: the PR's value
    against main's two runs."""
    table = {}
    for name, data in campaigns.items():
        for request in data["requests"]:
            key = f"{data['kind']}:{request['label']}"
            for metric in ("peakActiveMlxBytes", "peakFootprintBytes", "settledFootprintBytes"):
                table.setdefault(key, {}).setdefault(metric, {}).setdefault(data["arm"], []).append(
                    request[metric])
    rows = {}
    for key, metrics in table.items():
        rows[key] = {}
        for metric, arms in metrics.items():
            main = [v for v in arms.get("main", []) if v is not None]
            pr = [v for v in arms.get("pr", []) if v is not None]
            if not main or not pr:
                rows[key][metric] = {"main": main, "pr": pr, "verdict": "incomplete"}
                continue
            rows[key][metric] = {
                "main": main, "pr": pr, "mainSpread": max(main) - min(main),
                "prBelowEveryMain": max(pr) < min(main),
                "prWithinMainRange": max(pr) <= max(main),
            }
    return rows


def main():
    results = pathlib.Path(sys.argv[1])
    campaigns = dict(campaign(d) for d in sorted(results.iterdir())
                     if d.is_dir() and d.name.startswith("run") and not d.name.endswith("-ssd"))
    extras = {}
    for name in ("pr-parity", "pr-e2e"):
        outcome = load(results / name / "outcome.json")
        if outcome is not None:
            extras[name] = {k: v for k, v in outcome.items() if not k.endswith("UTC")}
    json.dump({"campaigns": campaigns, "prOnly": extras, "memoryComparison": compare(campaigns)},
              sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
