#!/usr/bin/env python3
"""Summarise one #469 arm from `ssd_writer_comparison.py`'s raw records.

For every request: the `admittingLeaf` instant (driver request start + the
sample's `elapsedMs`), each accepted `ssdAdmit` with its `storageRefCommit`
arrival, the driver-observed enqueue-to-commit interval, the segment file's
first sighting and final-size sample, and the footprint window from the
last sample before `admittingLeaf` to the commit arrival. Eviction and
deferral events are listed so the eviction stage's outcome is visible.
Prints JSON; pass --write to store it as `summary.json` in the arm's directory.
"""

import argparse
import json
import pathlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    d = args.directory
    requests = json.loads((d / "requests.json").read_text())
    events = json.loads((d / "events.json").read_text())
    samples = json.loads((d / "os-samples.json").read_text())
    files = json.loads((d / "file-samples.json").read_text())
    outcome = json.loads((d / "outcome.json").read_text())
    environment = json.loads((d / "environment.json").read_text())
    summary = {"arm": outcome["arm"], "build": environment["build"], "outcome": outcome, "requests": []}
    for request in requests:
        start = request["beginMonotonicSeconds"]
        own = [e for e in request["events"] if e.get("requestID") == request["requestID"]]
        admitting = [e for e in own if e["eventName"] == "requestMemory" and e["fields"].get("phase") == "admittingLeaf"]
        admit_at = start + float(admitting[0]["fields"]["elapsedMs"]) / 1000 if admitting else None
        capturing = [e for e in own if e["eventName"] == "requestMemory" and e["fields"].get("phase") == "capturingLeaf"]
        capture_at = start + float(capturing[0]["fields"]["elapsedMs"]) / 1000 if capturing else None
        record = {"label": request["label"], "promptTokens": (request["usage"] or {}).get("prompt_tokens"),
                  "responseSeconds": round(request["responseSeconds"], 2),
                  "admittingLeafMonotonicSeconds": admit_at, "capturingLeafMonotonicSeconds": capture_at,
                  "writes": [], "evictions": [], "deferrals": []}
        window_events = request["events"]
        for admit in [e for e in window_events if e["eventName"] == "ssdAdmit"]:
            snapshot_id = admit["fields"]["id"]
            commit = next((e for e in window_events if e["eventName"] == "storageRefCommit" and e["fields"].get("id") == snapshot_id), None)
            prepare = next((e for e in window_events if e["eventName"] == "ssdPayloadPrepare" and e["fields"].get("id") == snapshot_id), None)
            sightings = [f for f in files if snapshot_id in f["file"]]
            first_seen = sightings[0]["monotonicSeconds"] if sightings else None
            final = sightings[-1] if sightings else None
            commit_at = commit["arrivalMonotonicSeconds"] if commit else None
            write = {"snapshotID": snapshot_id, "bytes": int(admit["fields"].get("bytes", 0)),
                     "outcome": admit["fields"].get("outcome"), "writeClass": admit["fields"].get("writeClass"),
                     "inProcessWriteMs": admit["fields"].get("writeMs"),
                     "inProcessEnqueueToCommitMs": admit["fields"].get("enqueueToCommitMs"),
                     "payloadPrepareMs": prepare["fields"].get("durationMs") if prepare else None,
                     "admitArrivalMonotonicSeconds": admit["arrivalMonotonicSeconds"],
                     "commitArrivalMonotonicSeconds": commit_at,
                     "driverEnqueueToCommitSeconds": (commit_at - admit_at) if (commit_at and admit_at) else None,
                     "fileFirstSeenMonotonicSeconds": first_seen,
                     "fileFinalBytes": final["bytes"] if final else None,
                     "fileFinalSizeMonotonicSeconds": final["monotonicSeconds"] if final else None,
                     "fileWriteSeconds": (final["monotonicSeconds"] - first_seen) if (final and first_seen) else None,
                     "fileFirstSeenAfterAdmittingSeconds": (first_seen - admit_at) if (first_seen and admit_at) else None}
            if admit_at and commit_at:
                before = [s for s in samples if s["monotonicSeconds"] < admit_at]
                during = [s for s in samples if admit_at <= s["monotonicSeconds"] <= commit_at + 0.25]
                baseline_footprint = before[-1]["processFootprintBytes"] if before else None
                peak = max((s["processFootprintBytes"] for s in during), default=None)
                write["footprintBeforeAdmittingBytes"] = baseline_footprint
                write["peakFootprintDuringWriteBytes"] = peak
                write["footprintDeltaDuringWriteBytes"] = (peak - baseline_footprint) if (peak is not None and baseline_footprint is not None) else None
                write["samplesDuringWrite"] = len(during)
            record["writes"].append(write)
        for e in window_events:
            if e["eventName"] in ("eviction", "ssdEvictAtAdmission"):
                record["evictions"].append({"event": e["eventName"], "fields": e["fields"], "arrival": e["arrivalMonotonicSeconds"]})
            if e["eventName"] in ("ssdWriteDeferred", "ssdWritePromoted"):
                record["deferrals"].append({"event": e["eventName"], "fields": e["fields"]})
        stage_samples = [s for s in samples if s["stage"] in (request["label"], request["label"] + ":settle")]
        record["peakFootprintStageBytes"] = max((s["processFootprintBytes"] for s in stage_samples), default=None)
        record["settledFootprintBytes"] = request["settledFootprintBytes"]
        summary["requests"].append(record)
    text = json.dumps(summary, indent=2)
    if args.write:
        (d / "summary.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
