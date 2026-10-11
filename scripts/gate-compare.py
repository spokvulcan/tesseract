#!/usr/bin/env python3
"""Compare two gate runs (scripts/gate.sh run folders): did a change keep
the server's behaviour?

Usage: scripts/gate-compare.py <before-run> <after-run> [--show N]

Four comparisons, the bar ADR-0078 decision 8 sets for a refactor:

  checks       the report's check lines (✅/❌), timings stripped
  decode       the report's decode and acceptance lines, timings stripped
  diagnostics  the run's CacheDiagnostics, ids and timings masked. Lines
               written from the request's own task compare in order, per
               request; lines another task writes (the SSD writer, the
               memory observer, the Speculative Canonical Prefill) compare
               as a multiset per request, since their place follows the
               scheduler. The lines that follow the clock or free memory
               (periodic memory samples, budget re-measures) are dropped.
  traces       each Completion Trace Log record's restored offset, hit
               tokens and rewind, in order

Exits 0 when the checks, decode lines, in-order diagnostics and traces all
match, 1 otherwise. Other-task differences print but don't fail: they move
with the machine's load (the Speculative Canonical Prefill's `not-idle`
skips, a memory sample landing after the run's last request).
"""

import collections
import difflib
import json
import pathlib
import re
import sys

UUID = re.compile(r"[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}")
TIMING_KEY = re.compile(r"(Ms|Seconds|Secs|Time)$")
VOLATILE_KEYS = {
    "activeMlxBytes", "cachedMlxBytes", "peakMlxBytes", "mlxCacheLimitBytes",
    "processFootprintBytes", "processResidentBytes", "processCompressedBytes",
    "processLifetimePeakMlxBytes", "sampledRequestPeakActiveMlxBytes",
    "sampledRequestPeakFootprintBytes", "systemSwapUsedBytes",
    "sequence",  # per-request sample counter: periodic samples interleave
    "observedUnixSeconds", "freeDiskBytes", "availableBytes",
    "ceilingBytes", "headroomBytes", "budgetBytes", "treeBudgetBytes",
    "previousBytes", "currentBytes", "reclaimableBytes", "workingSetBytes",
    "measuredHeadroomBytes", "ssdBudgetBytes", "diskFreeBytes",
    # Eviction scores read the clock (recency).
    "normalizedRecency", "utility",
    # The SSD writer's progress, read from the request's task.
    "ssdPendingPayloadBytes", "ssdPendingPayloadCount",
    # A segment file's JSON header carries the descriptor's dates.
    "fileBytes",
}
# Fields another task's timing decides inside a request's own line: the
# leafStore report folds the request's Emitted Path resolves, and the
# Speculative Canonical Prefill's detached probe adds its own when it lands
# first.
RACY_FIELDS = {("leafStore", "emittedPathResolves"), ("leafStore", "emittedPathHits")}
DROP_EVENTS = {"budgetMeasure", "budgetChange"}
OTHER_TASK_EVENTS = {"ssdAdmit", "storageRefCommit", "ssdPayloadPrepare", "ssdWriteDeferred"}
REPORT_PREFIX = re.compile(r"^\[[^]]*\] *")
REPORT_TIMINGS = re.compile(r"ttftB/ttftA=[0-9.]+|ttft[A-Za-z0-9]*=[0-9.]+s?")


def report_lines(run):
    reports = sorted(run.glob("bench/*/latest.log"))
    if not reports:
        return None
    lines = [REPORT_TIMINGS.sub("", REPORT_PREFIX.sub("", line.rstrip("\n")))
             for line in reports[0].open()]
    checks = [line for line in lines if "✅" in line or "❌" in line]
    decode = [line for line in lines if "decode=" in line and "✅" not in line]
    overall = next((line for line in lines if "Overall" in line), "")
    return checks, decode, overall


def telemetry_files(run, component):
    """A component's files in write order: each day's rotated `.old` first."""
    files = []
    for day in sorted((run / "telemetry" / component).glob("*.jsonl")):
        old = day.with_name(day.name + ".old")
        if old.exists():
            files.append(old)
        files.append(day)
    return files


def diagnostics(run):
    aliases = {}

    def alias(kind, value):
        key = (kind, value)
        if key not in aliases:
            aliases[key] = f"<{kind}{sum(1 for k in aliases if k[0] == kind) + 1}>"
        return aliases[key]

    ordered = collections.defaultdict(list)
    unordered = collections.defaultdict(list)
    requests = []
    dropped = collections.Counter()
    for path in telemetry_files(run, "CacheDiagnostics"):
        for raw in path.open():
            event = json.loads(raw)
            name = event["eventName"]
            fields = {f["key"]: f["value"] for f in event.get("fields", [])}
            if name in DROP_EVENTS:
                dropped[name] += 1
                continue
            if name == "requestMemory" and fields.get("sampleKind") == "periodic":
                dropped["requestMemory/periodic"] += 1
                continue
            masked = []
            for key in sorted(fields):
                value = fields[key] or ""
                if key in VOLATILE_KEYS or TIMING_KEY.search(key) or (name, key) in RACY_FIELDS:
                    value = "~"
                else:
                    value = UUID.sub(lambda m: alias("id", m.group(0)), value)
                masked.append(f"{key}={value}")
            request = event.get("requestID")
            request = alias("req", request) if request else "-"
            if request != "-" and request not in requests:
                requests.append(request)
            line = f"{event.get('scope')} {name} {event.get('modelID')} " + " ".join(masked)
            other_task = (
                event.get("scope") == "system"
                or fields.get("sampleKind") == "afterRelease"
                or fields.get("stage") == "speculativePrefill"
                # The Speculative Canonical Prefill's probes resolve on its task.
                or fields.get("spelling") == "admissionProbe"
                or name in OTHER_TASK_EVENTS
            )
            (unordered if other_task else ordered)[request].append(line)
    return ordered, unordered, requests, dropped


def traces(run):
    records = []
    for path in telemetry_files(run, "PrefixCacheTraces"):
        for raw in path.open():
            try:
                record = json.loads(raw).get("record")
            except json.JSONDecodeError:
                continue
            if record:
                records.append((
                    record.get("modelID"), record.get("restoredOffset"),
                    record.get("hitTokens"), record.get("rewind") is not None,
                ))
    return records


def compare_lines(label, before, after, show):
    if before == after:
        print(f"{label}: identical ({len(before)} lines)")
        return True
    print(f"{label}: differ")
    for line in list(difflib.unified_diff(before, after, lineterm="", n=0))[2:2 + show]:
        print("   " + line[:400])
    return False


def main():
    args = sys.argv[1:]
    show = 40
    if "--show" in args:
        at = args.index("--show")
        show = int(args[at + 1])
        del args[at:at + 2]
    if len(args) != 2:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    before, after = (pathlib.Path(a) for a in args)
    same = True

    reports = report_lines(before), report_lines(after)
    if None in reports:
        print("report: missing on one side")
        same = False
    else:
        for run, (checks, _, overall) in zip((before, after), reports):
            passed = sum("✅" in line for line in checks)
            failed = sum("❌" in line for line in checks)
            print(f"{run.name}: {passed} passed, {failed} failed  {overall}")
        same &= compare_lines("checks", reports[0][0], reports[1][0], show)
        same &= compare_lines("decode", reports[0][1], reports[1][1], show)

    b_ordered, b_unordered, b_requests, b_dropped = diagnostics(before)
    a_ordered, a_unordered, a_requests, a_dropped = diagnostics(after)
    count = lambda lines: sum(len(v) for v in lines.values())
    print(f"diagnostics: in-order {count(b_ordered)} / {count(a_ordered)} lines, "
          f"other-task {count(b_unordered)} / {count(a_unordered)}, "
          f"requests {len(b_requests)} / {len(a_requests)}, same order {b_requests == a_requests}; "
          f"dropped {dict(b_dropped)} / {dict(a_dropped)}")
    same &= b_requests == a_requests
    in_order = 0
    printed = 0
    for request in sorted(set(b_ordered) | set(a_ordered), key=lambda r: (len(r), r)):
        b, a = b_ordered.get(request, []), a_ordered.get(request, [])
        if b == a:
            continue
        diff = list(difflib.unified_diff(b, a, lineterm="", n=0))
        changed = [d for d in diff if d[:1] in "+-" and not d.startswith(("+++", "---"))]
        in_order += len(changed)
        if printed < show:
            print(f"== in-order difference in {request}: {len(changed)} lines")
            for line in diff[2:2 + show - printed]:
                print("   " + line[:600])
                printed += 1
    print(f"diagnostics in-order differing lines: {in_order}")
    same &= in_order == 0
    other = 0
    for request in sorted(set(b_unordered) | set(a_unordered), key=lambda r: (len(r), r)):
        b = collections.Counter(b_unordered.get(request, []))
        a = collections.Counter(a_unordered.get(request, []))
        only_b, only_a = b - a, a - b
        if not (only_b or only_a):
            continue
        other += sum(only_b.values()) + sum(only_a.values())
        if printed < show:
            print(f"== other-task difference in {request}")
            for sign, lines in (("-", only_b), ("+", only_a)):
                for line in list(lines.elements())[:5]:
                    print(f"   {sign} " + line[:600])
                    printed += 1
    print(f"diagnostics other-task differing lines: {other} (scheduling; not a failure)")

    b_traces, a_traces = traces(before), traces(after)
    print(f"traces: {len(b_traces)} / {len(a_traces)} records")
    if b_traces != a_traces:
        same = False
        for at, (b, a) in enumerate(zip(b_traces, a_traces)):
            if b != a:
                print(f"   record {at}: {b} → {a}")

    print("SAME" if same else "DIFFERENT")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
