#!/usr/bin/env python3
"""#469 matched SSD-writer comparison: one build, one arm of the plan.

Launches an isolated Tesseract Agent (the binary the plan names for this
arm) with its HTTP server, a fresh SSD prefix-cache root under the output
directory, and the plan's RAM budget cap. It then runs the plan's
conversations: each `leaf` stage is a single ~3 GB turn whose end-of-turn
leaf is a guarantee-class SSD write; the `demotion` stages fill RAM past
the cap so an unbacked body is demoted (Recoverable Eviction) instead of
dropped. Every SSD write is observed three ways so the baseline build,
which reports no in-process timing, is measured the same way as the new
one: (1) the CacheDiagnostics sink is tailed every 20 ms and each line's
arrival time recorded; (2) the SSD root is polled every 50 ms for segment
files appearing and growing; (3) the OS is sampled every 250 ms for the
process footprint, with the plan's stop thresholds enforced and no retry.
Loaded-model use is under the owner's standing approval; the plan file is
committed before the run.
"""

import argparse
import ctypes
import datetime
import http.client
import json
import os
import pathlib
import socket
import subprocess
import threading
import time

from allocation_inventory_probe import OSProbe, file_digest, write_json
from warm_body_parity import available_bytes

EVENTS = {"requestMemory", "leafStore", "capture", "lookup", "ssdAdmit", "ssdPayloadPrepare",
          "storageRefCommit", "ssdWriteDeferred", "ssdWritePromoted", "eviction",
          "ssdEvictAtAdmission", "leafSupersession", "leafExtensionCommit", "budgetChange",
          "budgetMeasure"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, required=True)
    parser.add_argument("--arm", required=True, help="arm name from the plan's builds table")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("status") != "APPROVED":
        parser.error("Plan status is not APPROVED")
    for key, value in plan["resources"].items():
        if value is None:
            parser.error(f"Plan resource {key} is null: not executable")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(args.plan)], capture_output=True).returncode:
        parser.error("Commit the plan before running")
    if args.output.exists():
        parser.error("Refusing to overwrite an evidence directory")
    if subprocess.run(["pgrep", "-x", "Tesseract Agent"], capture_output=True).returncode == 0:
        parser.error("Quit the existing app before this isolated campaign")
    build = plan["builds"][args.arm]
    app = pathlib.Path(build["binary"])
    if file_digest(app) != build["binarySHA256"]:
        parser.error("Binary SHA-256 differs from the plan for arm " + args.arm)
    port = plan["port"]
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", port))
    resources = plan["resources"]
    args.output.mkdir(parents=True)
    args.output = args.output.resolve()
    ssd_root = args.output / "ssd-root"
    ssd_root.mkdir()

    probe = OSProbe()
    page_size = probe.sysctl("vm.pagesize", ctypes.c_uint32()).value
    before = probe.read(os.getpid())
    if before["pressureLevel"] != 1:
        parser.error("Memory pressure is not normal")
    initial_available = available_bytes(probe, page_size)
    if initial_available < resources["minimumInitialAvailableBytes"]:
        parser.error("Initial available memory below the plan minimum")
    disk_free = os.statvfs(str(args.output)).f_bavail * os.statvfs(str(args.output)).f_frsize
    if disk_free < resources["minimumInitialDiskFreeBytes"]:
        parser.error("Initial free disk below the plan minimum")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    launch = ["-serverPort", str(port), "-isServerEnabled", "YES",
              "-prefixCacheSSDEnabled", "YES",
              "-prefixCacheSSDDirectoryOverride", str(ssd_root),
              "-prefixCacheRAMBudgetCapBytes", str(plan["ramBudgetCapBytes"])]
    write_json(args.output / "environment.json", {
        "plan": plan, "planSHA256": file_digest(args.plan), "arm": args.arm, "build": build,
        "sourceRevision": head,
        "workingTreeDirty": subprocess.run(["git", "diff", "--quiet", "HEAD"], capture_output=True).returncode != 0,
        "binarySHA256": file_digest(app), "launchArguments": launch,
        "os": subprocess.check_output(["sw_vers"], text=True),
        "physicalMemoryBytes": probe.sysctl("hw.memsize", ctypes.c_uint64()).value,
        "initialAvailableBytes": initial_available, "initialDiskFreeBytes": disk_free,
        "baselineSystemSwapUsedBytes": before["systemSwapUsedBytes"],
        "startUTC": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    model_root = pathlib.Path.home() / "Library/Application Support/models"
    files = []
    for folder in plan["modelFolders"]:
        for path in sorted((model_root / folder).iterdir()):
            if path.is_file() and (path.suffix == ".safetensors" or path.suffix in (".json", ".jinja")):
                files.append({"modelFile": str(path.relative_to(model_root)), "bytes": path.stat().st_size,
                              "sha256": file_digest(path)})
    write_json(args.output / "model-files.json", files)
    for name in ("ssd_writer_comparison.py", "allocation_inventory_probe.py", "warm_body_parity.py"):
        (args.output / name).write_bytes(pathlib.Path(__file__).with_name(name).read_bytes())

    diagnostics = pathlib.Path.home() / "Library/Application Support/CacheDiagnostics"
    day = datetime.datetime.now().astimezone().strftime("%Y-%m-%d")
    diagnostic_path = diagnostics / (day + ".jsonl")
    offset = diagnostic_path.stat().st_size if diagnostic_path.exists() else 0
    identity = diagnostic_path.stat().st_ino if diagnostic_path.exists() else None
    partial = b""
    events, samples, results, attempts, abort_reason, rotations, file_samples = [], [], [], [], [], [], []
    aborted, finished = threading.Event(), threading.Event()
    active_socket, active_request_started, stage = [None], [None], ["launch"]
    started = time.monotonic()
    app_log = (args.output / "app.log").open("wb")
    child = subprocess.Popen([str(app), *launch], stdout=app_log, stderr=subprocess.STDOUT,
                             env={**os.environ, "TESSERACT_ALLOCATION_DIAGNOSTICS": "1"})

    def cancel_transport():
        sock = active_socket[0]
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def watchdog():
        while not finished.wait(resources["sampleIntervalMilliseconds"] / 1000):
            if child.poll() is not None:
                return
            try:
                sample = probe.read(child.pid)
                sample["availableBytes"] = available_bytes(probe, page_size)
                sample["stage"] = stage[0]
                sample["monotonicSeconds"] = time.monotonic()
                samples.append(sample)
                reason = None
                if sample["processFootprintBytes"] >= resources["footprintStopBytes"]:
                    reason = "process footprint stop"
                elif sample["availableBytes"] < resources["minimumAvailableStopBytes"]:
                    reason = "available memory stop"
                elif sample["systemSwapUsedBytes"] - before["systemSwapUsedBytes"] >= resources["swapGrowthStopBytes"]:
                    reason = "system swap growth stop"
                elif sample["pressureLevel"] >= resources["pressureStopLevel"] or sample["pressureLevel"] < 1:
                    reason = "memory pressure stop"
                elif time.monotonic() - started >= resources["maximumCampaignSeconds"]:
                    reason = "campaign time stop"
                elif (active_request_started[0] is not None and
                      time.monotonic() - active_request_started[0] >= resources["maximumRequestSeconds"]):
                    reason = "request time stop"
                elif os.statvfs(str(args.output)).f_bavail * os.statvfs(str(args.output)).f_frsize < resources["minimumDiskFreeStopBytes"]:
                    reason = "free disk stop"
            except OSError as error:
                reason = "OS sampling failed: " + str(error)
            if reason:
                abort_reason.append(reason)
                aborted.set()
                cancel_transport()
                if not finished.wait(resources["cancelGraceSeconds"]) and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                return

    def file_watcher():
        # Segment files appear when the writer opens them and grow to their
        # final size; the manifest rewrite follows the commit. Sizes are
        # recorded only on change, with a monotonic time.
        known = {}
        while not finished.wait(0.05):
            now = time.monotonic()
            try:
                for path in ssd_root.rglob("*"):
                    if not path.is_file():
                        continue
                    size = path.stat().st_size
                    key = str(path.relative_to(ssd_root))
                    if known.get(key) != size:
                        known[key] = size
                        file_samples.append({"monotonicSeconds": now, "file": key, "bytes": size,
                                             "stage": stage[0]})
            except OSError:
                pass

    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()
    file_thread = threading.Thread(target=file_watcher, daemon=True)
    file_thread.start()

    def collect_events():
        nonlocal offset, identity, partial
        if not diagnostic_path.exists():
            return
        stat = diagnostic_path.stat()
        raw = b""
        if identity is not None and (stat.st_ino != identity or stat.st_size < offset):
            old_path = diagnostic_path.with_suffix(".jsonl.old")
            if not old_path.exists() or old_path.stat().st_ino != identity:
                raise RuntimeError("Diagnostics rotated without a readable predecessor; attribution lost")
            with old_path.open("rb") as source:
                source.seek(offset)
                raw += source.read()
            offset = 0
            rotations.append(time.monotonic())
        identity = stat.st_ino
        with diagnostic_path.open("rb") as source:
            source.seek(offset)
            raw += source.read()
            offset = source.tell()
        arrival = time.monotonic()
        lines = (partial + raw).split(b"\n")
        partial = lines.pop()
        for line in lines:
            if not line:
                continue
            event = json.loads(line)
            if event.get("eventName") in EVENTS:
                event["fields"] = {field["key"]: field["value"] for field in event["fields"]}
                event["arrivalMonotonicSeconds"] = arrival
                event["stage"] = stage[0]
                events.append(event)

    def pause(seconds, poll=0.02):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if aborted.is_set():
                raise RuntimeError(abort_reason[-1])
            if child.poll() is not None:
                raise RuntimeError("Validation app exited")
            collect_events()
            time.sleep(poll)

    def request(label, messages, max_tokens, affinity):
        stage[0] = label
        begin_events = len(events)
        payload = {"model": plan["model"], "messages": messages, "max_tokens": max_tokens,
                   "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                   "reasoning_effort": plan["reasoningEffort"],
                   "chat_template_kwargs": {"preserve_thinking": False}}
        encoded = json.dumps(payload).encode()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=resources["maximumRequestSeconds"])
        start = time.monotonic()
        attempts.append({"label": label, "requestBytes": len(encoded), "beginMonotonicSeconds": start})
        write_json(args.output / "request-attempts.json", attempts)
        active_request_started[0] = start
        content, usage, first_delta, done, generated = [], None, None, False, 0
        try:
            conn.request("POST", "/v1/chat/completions", body=encoded,
                         headers={"Content-Type": "application/json", "x-session-affinity": affinity})
            active_socket[0] = conn.sock.dup()
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError("HTTP status " + str(response.status))
            while line := response.readline():
                if not line.startswith(b"data: "):
                    continue
                if line.strip() == b"data: [DONE]":
                    done = True
                    break
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    raise RuntimeError("Streaming error")
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("content") or delta.get("reasoning_content"):
                        generated += 1
                        first_delta = first_delta or time.monotonic()
        finally:
            conn.close()
            if active_socket[0]:
                active_socket[0].close()
            active_socket[0] = None
            active_request_started[0] = None
        response_end = time.monotonic()
        if not done:
            raise RuntimeError("Response ended without DONE")
        stage[0] = label + ":settle"
        # Wait until every SSD admission this request enqueued has either
        # committed or been reported rejected/deferred, then the settle pad.
        deadline = time.monotonic() + resources["maximumWriteSettleSeconds"]
        request_events = []
        while time.monotonic() < deadline:
            pause(0.2)
            request_events = [e for e in events[begin_events:] if e["eventName"] == "requestMemory"]
            if len({e["requestID"] for e in request_events}) > 1:
                raise RuntimeError("Overlapping HTTP requests; attribution invalid")
            released = any(e["fields"].get("sampleKind") == "afterRelease" for e in request_events)
            admitted = {e["fields"]["id"] for e in events[begin_events:]
                        if e["eventName"] == "ssdAdmit" and e["fields"].get("outcome") == "accepted"}
            committed = {e["fields"]["id"] for e in events[begin_events:] if e["eventName"] == "storageRefCommit"}
            if released and admitted and admitted <= committed:
                break
        else:
            raise RuntimeError("SSD writes for the request did not commit within the settle window")
        pause(plan["settleSeconds"])
        collect_events()
        request_id = request_events[0]["requestID"]
        related = [e for e in events[begin_events:]]
        settled = probe.read(child.pid)
        record = {"label": label, "requestID": request_id, "usage": usage, "generatedDeltas": generated,
                  "done": done, "beginMonotonicSeconds": start, "responseSeconds": response_end - start,
                  "firstDeltaSeconds": first_delta - start if first_delta else None,
                  "settledFootprintBytes": settled["processFootprintBytes"],
                  "events": related}
        results.append(record)
        write_json(args.output / "requests.json", results)
        write_json(args.output / "file-samples.json", file_samples)
        print(json.dumps({"completed": label, "usage": usage, "generated": generated,
                          "seconds": round(record["responseSeconds"], 1),
                          "footprintGB": round(settled["processFootprintBytes"] / 1e9, 2)}), flush=True)
        return {"role": "assistant", "content": "".join(content)}

    failure = None
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if aborted.is_set():
                raise RuntimeError(abort_reason[-1])
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                conn.request("GET", "/v1/models")
                models = json.loads(conn.getresponse().read())
                conn.close()
                if any(m["id"] == plan["model"] and m.get("state") in ("loaded", "available") for m in models["data"]):
                    write_json(args.output / "models.json", models)
                    break
            except (OSError, ValueError, http.client.HTTPException):
                pass
            pause(0.5, poll=0.1)
        else:
            raise RuntimeError("Model was not available before the startup deadline")
        stage[0] = "server-ready-idle"
        pause(5, poll=0.1)
        filler = pathlib.Path(plan["fillerFile"]).read_text()
        if file_digest(pathlib.Path(plan["fillerFile"])) != plan["fillerSHA256"]:
            raise RuntimeError("Filler corpus differs from the plan")
        system = {"role": "system", "content": "You are a careful assistant answering questions about the document the user pastes. Be thorough."}
        conversations = {}
        for step in plan["steps"]:
            conversation = conversations.setdefault(step["conversation"], [system])
            if step["kind"] == "document":
                chunk, used = "", step["fillerOffset"]
                while len(chunk) < step["characters"]:
                    start_at = used % len(filler)
                    piece = filler[start_at:start_at + step["characters"] - len(chunk)]
                    chunk += piece + ("\n\n" if start_at + len(piece) >= len(filler) else "")
                    used += len(piece)
                conversation.append({"role": "user", "content": f"Here is a document:\n\n{chunk}\n\nIn one sentence, what is this document about?"})
            else:
                conversation.append({"role": "user", "content": "Name one term from the document."})
            answer = request(step["id"], conversation, plan["shortOutputTokens"], "writer-469-" + step["conversation"])
            conversation.append(answer)
        stage[0] = "final-idle"
        pause(10, poll=0.1)
    except Exception as error:
        failure = abort_reason[-1] if aborted.is_set() and abort_reason else str(error)
    finally:
        try:
            collect_events()
        except Exception as error:
            failure = failure or str(error)
        finished.set()
        watcher.join(timeout=2)
        file_thread.join(timeout=2)
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        app_log.close()
        write_json(args.output / "os-samples.json", samples)
        write_json(args.output / "events.json", events)
        write_json(args.output / "file-samples.json", file_samples)
        manifest = ssd_root / "manifest.json"
        outcome = {"failure": failure, "arm": args.arm, "exitCode": child.returncode,
                   "diagnosticsRotations": len(rotations), "elapsedSeconds": time.monotonic() - started,
                   "sampleCount": len(samples),
                   "maxSampledFootprintBytes": max((s["processFootprintBytes"] for s in samples), default=None),
                   "minSampledAvailableBytes": min((s["availableBytes"] for s in samples), default=None),
                   "maxPressureLevel": max((s["pressureLevel"] for s in samples), default=None),
                   "maxSystemSwapGrowthBytes": max((s["systemSwapUsedBytes"] - before["systemSwapUsedBytes"] for s in samples), default=None),
                   "ssdRootBytes": sum(p.stat().st_size for p in ssd_root.rglob("*") if p.is_file()),
                   "manifestPresent": manifest.exists(),
                   "endUTC": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(args.output / "outcome.json", outcome)
        print(json.dumps(outcome, indent=2), flush=True)
        if failure:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
