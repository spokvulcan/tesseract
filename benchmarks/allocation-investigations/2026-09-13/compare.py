#!/usr/bin/env python3
"""Compare the preserved loading experiment; never launches a model."""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


captures = []
requests = {}
for name in ("preserve-cache", "clear-cache"):
    folder = ROOT / name
    summary = read(folder / "summary.json")
    phases = {e["phase"]: e for e in summary["allocationPhases"] if e["phase"].startswith("model")}
    begin = float(phases["modelLoadBegin"]["observedUnixSeconds"])
    end = float(phases["modelLoadCompleted"]["observedUnixSeconds"])
    rows = [r for r in read(folder / "os-samples.json") if begin <= r["unixSeconds"] <= end]
    record = {"capture": name, "binary": read(folder / "environment.json")["binarySHA256"],
              "loadPeakBytes": max(r["processFootprintBytes"] for r in rows),
              "loadSeconds": float(phases["modelLoadCompleted"]["loadSeconds"]),
              "targetStackSeconds": float(phases["modelTargetStackingEnd"]["observedUnixSeconds"])
              - float(phases["modelTargetStackingBegin"]["observedUnixSeconds"]),
              "os": summary["os"]}
    for key, phase, field in (
            ("activeAfterDraft", "modelDFlash2Loaded", "activeMlxBytes"),
            ("cachedAfterDraft", "modelDFlash2Loaded", "cachedMlxBytes"),
            ("activeBeforeStack", "modelTargetStackingBegin", "activeMlxBytes"),
            ("cachedBeforeStack", "modelTargetStackingBegin", "cachedMlxBytes"),
            ("footprintBeforeStack", "modelTargetStackingBegin", "processFootprintBytes"),
            ("activeAfterLoad", "modelLoadCompleted", "activeMlxBytes"),
            ("lifetimePeakAfterLoad", "modelLoadCompleted", "processLifetimePeakMlxBytes")):
        record[key] = int(phases[phase][field])
    captures.append(record)
    requests[name] = {r["label"]: r for r in read(folder / "requests.json")}

matches = []
for label, after in requests["clear-cache"].items():
    before = requests["preserve-cache"][label]
    matches.append({"label": label,
                    "sameRequest": before["requestSHA256"] == after["requestSHA256"],
                    "sameAssistantMessage": before["assistantMessageSHA256"] == after["assistantMessageSHA256"],
                    "sameUsage": before["usage"] == after["usage"]})

result = {"captures": captures, "requests": matches}
(ROOT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
assert captures[0]["binary"] == captures[1]["binary"]
assert all(m["sameRequest"] and m["sameAssistantMessage"] and m["sameUsage"] for m in matches)
print("Same binary; all seven requests, assistant messages and usage match.")
print("Sampled loading peaks:", [c["loadPeakBytes"] for c in captures])
