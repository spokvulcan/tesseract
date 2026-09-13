#!/usr/bin/env python3
"""Bounded production allocation capture for #506; explicit --run required.

Starts one Release app on a separate loopback port. Saves scalar diagnostics,
request/response hashes and 250 ms OS samples, never private conversation data.
Resource breaches end the campaign without retry. This is a measurement runner,
not an optimization benchmark or a bitwise cache-correctness gate.
"""

import argparse
import ctypes
import datetime
import hashlib
import http.client
import json
import os
import pathlib
import signal
import socket
import subprocess
import threading
import time

from capture_memory_replay import assistant_message

GIB = 1 << 30
EVENTS = {"requestMemory", "allocationMemory", "leafStore", "lookup",
          "ssdPayloadMaterialize", "leafLeaseBegin", "leafLeaseEnd", "leafRewind",
          "ssdAdmit", "storageRefCommit", "storageRefDropCallback", "leafExtensionCommit"}


class RUsage(ctypes.Structure):
    # SDK sys/resource.h: rusage_info_v4 starts with uuid + these eight uint64s.
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("values", ctypes.c_uint64 * 64)]


class Swap(ctypes.Structure):
    _fields_ = [("total", ctypes.c_uint64), ("available", ctypes.c_uint64),
                ("used", ctypes.c_uint64), ("page_size", ctypes.c_uint32),
                ("encrypted", ctypes.c_int)]


class OSProbe:
    def __init__(self):
        self.libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        self.libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

    def sysctl(self, name, value):
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if self.libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0):
            raise OSError(ctypes.get_errno(), "sysctl " + name)
        return value

    def read(self, pid):
        start = time.perf_counter_ns()
        usage = RUsage()
        if self.libproc.proc_pid_rusage(pid, 4, ctypes.byref(usage)):
            raise OSError(ctypes.get_errno(), "proc_pid_rusage")
        result = {"monotonicSeconds": time.monotonic(),
                  "unixSeconds": time.time(),
                  "processFootprintBytes": int(usage.values[7]),
                  "processResidentBytes": int(usage.values[6]),
                  "systemSwapUsedBytes": int(self.sysctl("vm.swapusage", Swap()).used),
                  "pressureLevel": int(self.sysctl("kern.memorystatus_vm_pressure_level",
                                                   ctypes.c_int()).value)}
        result["probeNanoseconds"] = time.perf_counter_ns() - start
        return result


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=18321)
    parser.add_argument("--allow-pressure-warning", action="store_true")
    parser.add_argument("--footprint-stop-gib", type=int, choices=(28, 32), default=28)
    parser.add_argument("--swap-growth-stop-gib", type=float, choices=(0.5, 2), default=0.5)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    allowed_pressure = (1, 2) if args.allow_pressure_warning else (1,)
    plan = {"footprintStopBytes": args.footprint_stop_gib * GIB,
            "swapGrowthStopBytes": int(args.swap_growth_stop_gib * GIB),
            "allowedPressureLevels": list(allowed_pressure), "sampleIntervalSeconds": 0.25,
            "maxCampaignSeconds": 900, "maxRequestSeconds": 180,
            "maxOutputTokens": 128, "settleSeconds": 5, "maxRequests": 7,
            "model": "qwen3.8-27b", "temperature": 0,
            "largeContextCampaign": False, "optimizationComparison": False}
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    if args.output.exists():
        parser.error("Refusing to overwrite an evidence directory")
    # A bound port or an existing app makes isolation unproven.
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", args.port))
    existing = subprocess.run(["pgrep", "-x", "Tesseract Agent"], capture_output=True)
    if existing.returncode == 0:
        parser.error("Quit the existing app before this isolated campaign")
    args.output.mkdir(parents=True)
    diagnostics = pathlib.Path.home() / "Library/Application Support/CacheDiagnostics"
    day = datetime.datetime.now().astimezone().strftime("%Y-%m-%d")
    diagnostic_path = diagnostics / (day + ".jsonl")
    offset = diagnostic_path.stat().st_size if diagnostic_path.exists() else 0
    identity = diagnostic_path.stat().st_ino if diagnostic_path.exists() else None
    partial = b""
    events = []
    samples = []
    results = []
    attempts = []
    aborted = threading.Event()
    finished = threading.Event()
    abort_reason = []
    active_connection = [None]
    active_socket = [None]
    active_request_started = [None]
    stage = ["launch"]
    started = time.monotonic()
    probe = OSProbe()
    before = probe.read(os.getpid())
    if before["pressureLevel"] != 1:
        parser.error("Memory pressure is not normal")
    metadata = {"plan": plan, "startUTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "sourceRevision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "sourceDiffSHA256": digest_bytes(subprocess.check_output(["git", "diff", "--", "tesseract"])),
                "binarySHA256": file_digest(args.app), "instrumentationEnabled": True,
                "baselineSystemSwapUsedBytes": before["systemSwapUsedBytes"],
                "os": subprocess.check_output(["sw_vers"], text=True),
                "physicalMemoryBytes": probe.sysctl("hw.memsize", ctypes.c_uint64()).value,
                "launchArguments": ["-serverPort", str(args.port), "-isServerEnabled", "YES"]}
    write_json(args.output / "environment.json", metadata)
    (args.output / "instrumentation.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--", "tesseract"]))
    runner = pathlib.Path(__file__).resolve()
    for source in (runner, runner.with_name("capture_memory_replay.py")):
        (args.output / source.name).write_bytes(source.read_bytes())
    model_root = pathlib.Path.home() / "Library/Application Support/models"
    files = []
    for folder in ("mlx-community_Qwen3.8-27B-4bit", "incoai_Qwen3.8-27B-DFlash2"):
        for path in sorted((model_root / folder).iterdir()):
            if path.is_file() and (path.suffix == ".safetensors" or path.name in
                                   {"config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"}):
                files.append({"modelFile": str(path.relative_to(model_root)),
                              "bytes": path.stat().st_size, "sha256": file_digest(path)})
    write_json(args.output / "model-files.json", files)
    app_log = open("/private/tmp/tesseract-506-production-app.log", "wb")
    child = subprocess.Popen([str(args.app), *metadata["launchArguments"]], stdout=app_log,
                             stderr=subprocess.STDOUT,
                             env={**os.environ, "TESSERACT_ALLOCATION_DIAGNOSTICS": "1"})

    def cancel_transport():
        sock = active_socket[0]
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def watchdog():
        while not finished.wait(0.25):
            if child.poll() is not None:
                return
            try:
                sample = probe.read(child.pid)
                sample["stage"] = stage[0]
                samples.append(sample)
                reason = None
                if sample["processFootprintBytes"] >= plan["footprintStopBytes"]:
                    reason = "process footprint stop"
                elif sample["systemSwapUsedBytes"] - before["systemSwapUsedBytes"] >= plan["swapGrowthStopBytes"]:
                    reason = "system swap growth stop"
                elif sample["pressureLevel"] not in allowed_pressure:
                    reason = "memory pressure stop"
                elif time.monotonic() - started >= plan["maxCampaignSeconds"]:
                    reason = "campaign time stop"
                elif (active_request_started[0] is not None and
                      time.monotonic() - active_request_started[0] >= plan["maxRequestSeconds"]):
                    reason = "request time stop"
            except OSError as error:
                reason = "OS sampling failed: " + str(error)
            if reason:
                abort_reason.append(reason)
                aborted.set()
                cancel_transport()
                if not finished.wait(5) and child.poll() is None:
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

    def request(label, messages, cancel_after=None):
        stage[0] = label
        begin_events = len(events)
        payload = {"model": plan["model"], "messages": messages,
                   "max_completion_tokens": plan["maxOutputTokens"], "temperature": 0,
                   "stream": True, "stream_options": {"include_usage": True}}
        encoded = json.dumps(payload).encode()
        if len(encoded) > 100_000:
            raise RuntimeError("Synthetic request size bound exceeded")
        conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=plan["maxRequestSeconds"])
        active_connection[0] = conn
        start = time.monotonic()
        attempts.append({"label": label, "requestSHA256": digest_bytes(encoded),
                         "requestBytes": len(encoded), "beginMonotonicSeconds": start,
                         "cancelAfterSeconds": cancel_after})
        write_json(args.output / "request-attempts.json", attempts)
        active_request_started[0] = start
        chunks, usage, first_delta, finish_delta = [], None, None, None
        timer = None
        done = False
        intentional_cancel = threading.Event()
        def cancel():
            intentional_cancel.set()
            cancel_transport()
        try:
            conn.request("POST", "/v1/chat/completions", body=encoded,
                         headers={"Content-Type": "application/json", "x-session-affinity": "allocation-506"})
            active_socket[0] = conn.sock.dup()
            if cancel_after is not None:
                timer = threading.Timer(cancel_after, cancel)
                timer.start()
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
                    chunks.append(delta)
                    if any(delta.get(key) for key in ("content", "reasoning_content", "tool_calls")):
                        first_delta = first_delta or time.monotonic()
                    if choice.get("finish_reason"):
                        finish_delta = time.monotonic()
        except (OSError, http.client.HTTPException):
            if not intentional_cancel.is_set():
                raise
        finally:
            if timer:
                timer.cancel()
            conn.close()
            if active_socket[0]:
                active_socket[0].close()
            active_socket[0] = None
            active_connection[0] = None
            active_request_started[0] = None
        response_end = time.monotonic()
        if not done and not intentional_cancel.is_set():
            raise RuntimeError("Response ended without DONE")
        stage[0] = label + ":settle"
        deadline = time.monotonic() + 60
        request_events = []
        while time.monotonic() < deadline:
            pause(0.2)
            request_events = [e for e in events[begin_events:] if e["eventName"] == "requestMemory"]
            ids = {e["requestID"] for e in request_events}
            if len(ids) > 1:
                raise RuntimeError("Overlapping HTTP requests; attribution invalid")
            if any(e["fields"].get("sampleKind") == "afterRelease" for e in request_events):
                break
        else:
            raise RuntimeError("Request release did not become observable")
        pause(plan["settleSeconds"])
        if not intentional_cancel.is_set():
            if not any(e["fields"].get("dflash2Engaged") == "true" for e in request_events):
                raise RuntimeError("DFlash2 engagement was not confirmed")
            if any(e.get("kvBits") is not None for e in request_events):
                raise RuntimeError("Expected unquantized KV")
        record = {"label": label, "requestID": request_events[0]["requestID"],
                  "requestSHA256": digest_bytes(encoded), "requestBytes": len(encoded),
                  "responseSHA256": digest_bytes(json.dumps(chunks, sort_keys=True).encode()),
                  "usage": usage, "cancelAfterSeconds": cancel_after,
                  "cancelled": intentional_cancel.is_set(), "done": done,
                  "responseSeconds": response_end - start,
                  "firstDeltaSeconds": first_delta - start if first_delta else None,
                  "finishDeltaToDoneSeconds": response_end - finish_delta if finish_delta else None,
                  "beginMonotonicSeconds": start, "responseEndMonotonicSeconds": response_end}
        results.append(record)
        write_json(args.output / "requests.json", results)
        if cancel_after is not None and not intentional_cancel.is_set():
            raise RuntimeError("Cancellation scenario completed before cancellation; not exercised")
        if intentional_cancel.is_set() and not any(
                e["fields"].get("sampleKind") == "cancelSignal" and
                e["fields"].get("cancelSignalOrigin") in ("caller", "streamCancelled")
                for e in request_events):
            raise RuntimeError("Client disconnect did not produce cancellation telemetry")
        print(json.dumps({"completed": label, "usage": usage, "seconds": record["responseSeconds"]}), flush=True)
        return assistant_message(chunks)

    failure = None
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if aborted.is_set():
                raise RuntimeError(abort_reason[-1])
            try:
                conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=2)
                conn.request("GET", "/v1/models")
                models = json.loads(conn.getresponse().read())
                conn.close()
                if any(m["id"] == plan["model"] and m.get("state") in ("loaded", "available")
                       for m in models["data"]):
                    write_json(args.output / "models.json", models)
                    break
            except (OSError, ValueError, http.client.HTTPException):
                pass
            pause(0.5)
        else:
            raise RuntimeError("Model was not available before the startup deadline")
        stage[0] = "server-ready-idle"
        pause(5)
        def text_rows(start, count):
            return "\n".join(f"Record {i:04d}: amber birch cedar dune elm fern grove hazel iris juniper."
                             for i in range(start, start + count))
        messages = [{"role": "system", "content": "You are a precise assistant for a synthetic memory measurement."},
                    {"role": "user", "content": text_rows(0, 140) + "\nDescribe the recurring pattern briefly."}]
        answer = request("cold", messages)
        messages += [answer, {"role": "user", "content": "Name three words from the records."}]
        answer = request("warm", messages)
        messages += [answer, {"role": "user", "content": text_rows(140, 140) + "\nDescribe these additional records briefly."}]
        answer = request("grow", messages)
        base = messages + [answer]
        growth = {"role": "user", "content": text_rows(280, 200) + "\nSummarize the full pattern."}
        request("cancel-growth", base + [growth], cancel_after=3.0)
        answer = request("resend-growth", base + [growth])
        messages = base + [growth, answer]
        for index in range(2):
            messages += [{"role": "user", "content": f"Give another short observation, number {index + 1}."}]
            answer = request(f"warm-repeat-{index + 1}", messages)
            messages += [answer]
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
        watcher.join(timeout=1)
        if child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        app_log.close()
        write_json(args.output / "os-samples.json", samples)
        with (args.output / "diagnostics.jsonl").open("w") as target:
            for event in events:
                target.write(json.dumps(event, sort_keys=True) + "\n")
        write_json(args.output / "outcome.json", {"failure": failure, "resourceStops": abort_reason,
                   "completedRequests": len(results), "appExitCode": child.returncode,
                   "elapsedSeconds": time.monotonic() - started})
    if failure:
        raise SystemExit(failure)


if __name__ == "__main__":
    main()
