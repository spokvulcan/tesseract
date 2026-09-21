#!/usr/bin/env python3
"""#501 profiling: retained attention capacity after a cancelled long generation.

Launches an isolated Tesseract Agent with its HTTP server, grows one
conversation to each registered prompt size, and at each size runs (a) a
long streaming generation that the client cancels after a fixed number of
generated tokens (Leaf Checkout → cancel → Leaf Rewind → check-in) and (b) an
ordinary short turn on the same conversation. Every `requestMemory`,
`leafRewind`, `leafStore`, `lookup` and lease event is collected from the
durable CacheDiagnostics sink and joined by request ID. The OS is sampled
every 250 ms; the campaign stops on the manifest's thresholds and is never
retried. Loaded-model use is under the owner's standing approval; the plan
file is committed before the run.
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

EVENTS = {"requestMemory", "leafRewind", "leafStore", "lookup", "leafLeaseBegin", "leafLeaseEnd",
          "leafSupersession", "warmCompress", "requestCancelled"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=pathlib.Path, required=True)
    parser.add_argument("--plan", type=pathlib.Path, required=True)
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
    port = plan["port"]
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", port))
    resources = plan["resources"]
    args.output.mkdir(parents=True)
    args.output = args.output.resolve()

    probe = OSProbe()
    page_size = probe.sysctl("vm.pagesize", ctypes.c_uint32()).value
    before = probe.read(os.getpid())
    if before["pressureLevel"] != 1:
        parser.error("Memory pressure is not normal")
    initial_available = available_bytes(probe, page_size)
    if initial_available < resources["minimumInitialAvailableBytes"]:
        parser.error("Initial available memory below the plan minimum")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    launch = ["-serverPort", str(port), "-isServerEnabled", "YES"]
    write_json(args.output / "environment.json", {
        "plan": plan, "planSHA256": file_digest(args.plan), "sourceRevision": head,
        "workingTreeDirty": subprocess.run(["git", "diff", "--quiet", "HEAD"], capture_output=True).returncode != 0,
        "binarySHA256": file_digest(args.app), "launchArguments": launch,
        "os": subprocess.check_output(["sw_vers"], text=True),
        "physicalMemoryBytes": probe.sysctl("hw.memsize", ctypes.c_uint64()).value,
        "initialAvailableBytes": initial_available,
        "baselineSystemSwapUsedBytes": before["systemSwapUsedBytes"],
        "startUTC": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    if file_digest(args.app) != plan["releaseBinarySHA256"]:
        parser.error("Release binary SHA-256 differs from the plan")
    model_root = pathlib.Path.home() / "Library/Application Support/models"
    files = []
    for folder in plan["modelFolders"]:
        for path in sorted((model_root / folder).iterdir()):
            if path.is_file() and (path.suffix == ".safetensors" or path.suffix in (".json", ".jinja")):
                files.append({"modelFile": str(path.relative_to(model_root)), "bytes": path.stat().st_size,
                              "sha256": file_digest(path)})
    write_json(args.output / "model-files.json", files)
    for name in ("cancelled_generation_profile.py", "allocation_inventory_probe.py", "warm_body_parity.py"):
        (args.output / name).write_bytes(pathlib.Path(__file__).with_name(name).read_bytes())

    diagnostics = pathlib.Path.home() / "Library/Application Support/CacheDiagnostics"
    day = datetime.datetime.now().astimezone().strftime("%Y-%m-%d")
    diagnostic_path = diagnostics / (day + ".jsonl")
    offset = diagnostic_path.stat().st_size if diagnostic_path.exists() else 0
    identity = diagnostic_path.stat().st_ino if diagnostic_path.exists() else None
    partial = b""
    events, samples, results, attempts, abort_reason = [], [], [], [], []
    aborted, finished = threading.Event(), threading.Event()
    active_socket, active_request_started, stage = [None], [None], ["launch"]
    started = time.monotonic()
    app_log = (args.output / "app.log").open("wb")
    child = subprocess.Popen([str(args.app), *launch], stdout=app_log, stderr=subprocess.STDOUT,
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

    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()

    def collect_events():
        nonlocal offset, identity, partial
        if not diagnostic_path.exists():
            return
        stat = diagnostic_path.stat()
        if identity is not None and (stat.st_ino != identity or stat.st_size < offset):
            raise RuntimeError("Diagnostics rotated; campaign stopped to preserve attribution")
        identity = stat.st_ino
        with diagnostic_path.open("rb") as source:
            source.seek(offset)
            raw = source.read()
            offset = source.tell()
        lines = (partial + raw).split(b"\n")
        partial = lines.pop()
        for line in lines:
            if not line:
                continue
            event = json.loads(line)
            if event.get("eventName") in EVENTS:
                event["fields"] = {field["key"]: field["value"] for field in event["fields"]}
                events.append(event)

    def pause(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if aborted.is_set():
                raise RuntimeError(abort_reason[-1])
            if child.poll() is not None:
                raise RuntimeError("Validation app exited")
            collect_events()
            time.sleep(0.1)

    def request(label, messages, max_tokens, cancel_after_tokens=None):
        stage[0] = label
        begin_events = len(events)
        payload = {"model": plan["model"], "messages": messages, "max_tokens": max_tokens,
                   "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                   "reasoning_effort": plan["reasoningEffort"],
                   "chat_template_kwargs": {"preserve_thinking": False}}
        encoded = json.dumps(payload).encode()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=resources["maximumRequestSeconds"])
        start = time.monotonic()
        attempts.append({"label": label, "requestBytes": len(encoded), "beginMonotonicSeconds": start,
                         "cancelAfterTokens": cancel_after_tokens})
        write_json(args.output / "request-attempts.json", attempts)
        active_request_started[0] = start
        content, reasoning, usage, first_delta, done, generated = [], [], None, None, False, 0
        intentional_cancel = threading.Event()
        cancel_at = None
        try:
            conn.request("POST", "/v1/chat/completions", body=encoded,
                         headers={"Content-Type": "application/json", "x-session-affinity": "profile-501"})
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
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    if delta.get("content") or delta.get("reasoning_content"):
                        generated += 1  # one delta per token on this server
                        first_delta = first_delta or time.monotonic()
                if cancel_after_tokens is not None and generated >= cancel_after_tokens:
                    intentional_cancel.set()
                    cancel_at = time.monotonic()
                    cancel_transport()
                    break
        except (OSError, http.client.HTTPException):
            if not intentional_cancel.is_set():
                raise
        finally:
            conn.close()
            if active_socket[0]:
                active_socket[0].close()
            active_socket[0] = None
            active_request_started[0] = None
        response_end = time.monotonic()
        if not done and not intentional_cancel.is_set():
            raise RuntimeError("Response ended without DONE")
        stage[0] = label + ":settle"
        deadline = time.monotonic() + 120
        request_events = []
        while time.monotonic() < deadline:
            pause(0.2)
            request_events = [e for e in events[begin_events:] if e["eventName"] == "requestMemory"]
            if len({e["requestID"] for e in request_events}) > 1:
                raise RuntimeError("Overlapping HTTP requests; attribution invalid")
            if any(e["fields"].get("sampleKind") == "afterRelease" for e in request_events):
                break
        else:
            raise RuntimeError("Request release did not become observable")
        pause(plan["settleSeconds"])
        collect_events()
        request_id = request_events[0]["requestID"]
        related = [e for e in events[begin_events:] if e.get("requestID") == request_id or e["eventName"] in ("leafRewind", "leafStore", "lookup", "leafLeaseBegin", "leafLeaseEnd")]
        settled = probe.read(child.pid)
        record = {"label": label, "requestID": request_id, "usage": usage, "generatedDeltas": generated,
                  "cancelled": intentional_cancel.is_set(), "done": done,
                  "responseSeconds": response_end - start,
                  "firstDeltaSeconds": first_delta - start if first_delta else None,
                  "cancelAtSeconds": cancel_at - start if cancel_at else None,
                  "settledFootprintBytes": settled["processFootprintBytes"],
                  "events": related}
        results.append(record)
        write_json(args.output / "requests.json", results)
        if cancel_after_tokens is not None and not intentional_cancel.is_set():
            raise RuntimeError("Cancellation scenario completed before cancellation; not exercised")
        if intentional_cancel.is_set():
            rewinds = [e for e in related if e["eventName"] == "leafRewind"]
            if not rewinds:
                raise RuntimeError("Cancelled request produced no Leaf Rewind")
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
            pause(0.5)
        else:
            raise RuntimeError("Model was not available before the startup deadline")
        stage[0] = "server-ready-idle"
        pause(5)
        filler = pathlib.Path(plan["fillerFile"]).read_text()
        if file_digest(pathlib.Path(plan["fillerFile"])) != plan["fillerSHA256"]:
            raise RuntimeError("Filler corpus differs from the plan")
        messages = [{"role": "system", "content": "You are a careful assistant answering questions about the document the user pastes. Be thorough."}]
        used = 0
        for step in plan["steps"]:
            chunk_chars = step["addCharacters"]
            chunk = ""
            while len(chunk) < chunk_chars:
                start_at = used % len(filler)
                piece = filler[start_at:start_at + chunk_chars - len(chunk)]
                chunk += piece + ("\n\n" if start_at + len(piece) >= len(filler) else "")
                used += len(piece)
            messages.append({"role": "user", "content": f"Here is more of the document:\n\n{chunk}\n\nIn one sentence, what is this part about?"})
            answer = request(f"{step['id']}-grow", messages, plan["shortOutputTokens"])
            messages.append(answer)
            long_user = {"role": "user", "content": "Write a very long, detailed, section-by-section commentary on everything in the document so far. Do not stop early."}
            request(f"{step['id']}-cancel", messages + [long_user], plan["longOutputTokens"],
                    cancel_after_tokens=step["cancelAfterTokens"])
            messages.append({"role": "user", "content": "Name one term from the document."})
            answer = request(f"{step['id']}-after", messages, plan["shortOutputTokens"])
            messages.append(answer)
        stage[0] = "final-idle"
        pause(10)
    except Exception as error:
        failure = abort_reason[-1] if aborted.is_set() and abort_reason else str(error)
    finally:
        try:
            collect_events()
        except Exception as error:
            failure = failure or str(error)
        finished.set()
        watcher.join(timeout=2)
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
        outcome = {"failure": failure, "exitCode": child.returncode, "elapsedSeconds": time.monotonic() - started,
                   "sampleCount": len(samples),
                   "maxSampledFootprintBytes": max((s["processFootprintBytes"] for s in samples), default=None),
                   "minSampledAvailableBytes": min((s["availableBytes"] for s in samples), default=None),
                   "maxPressureLevel": max((s["pressureLevel"] for s in samples), default=None),
                   "maxSystemSwapGrowthBytes": max((s["systemSwapUsedBytes"] - before["systemSwapUsedBytes"] for s in samples), default=None),
                   "endUTC": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(args.output / "outcome.json", outcome)
        print(json.dumps(outcome, indent=2), flush=True)
        if failure:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
