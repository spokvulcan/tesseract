#!/usr/bin/env python3
"""Replay one local request and retain scalar request-memory evidence (#478).

Run against an otherwise idle, isolated Tesseract HTTP server. Input may be
an HTTPRequestLogger recording (a // header followed by JSON). The output
contains hashes, counts, and diagnostics, never request/response text.
--next-request optionally writes a PRIVATE continuation fixture for a warm
turn; keep it outside the repository. Tool calls are echoed with a dummy
tool result and are never executed.

This is a bounded capture experiment, not the full historical session gate:
both builds must use identical request bytes and launch settings. It cannot
establish bitwise logit parity; use the loaded-model correctness runner too.
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import time
import urllib.parse
import urllib.request


EVENTS = {"requestMemory", "leafStore", "lookup", "ssdPayloadMaterialize"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_request(path):
    raw = path.read_text()
    if raw.startswith("//"):
        raw = raw.split("\n", 1)[1]
    return json.loads(raw)


def diagnostic_events(path, offset, identity):
    stat = path.stat()
    if (stat.st_dev, stat.st_ino) != identity or stat.st_size < offset:
        raise RuntimeError("Diagnostics rotated during replay; evidence is incomplete")
    with path.open() as source:
        source.seek(offset)
        events = []
        for line in source:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # The file sink may still be writing its last line.
                if not line.endswith("\n"):
                    break
                raise
            if event.get("eventName") in EVENTS:
                event["fields"] = {item["key"]: item["value"] for item in event["fields"]}
                events.append(event)
        return events


def await_release(path, offset, identity, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = diagnostic_events(path, offset, identity)
        samples = [e for e in events if e["eventName"] == "requestMemory"]
        request_ids = {e["requestID"] for e in samples}
        if len(request_ids) > 1:
            raise RuntimeError("Concurrent requests detected; repeat on an isolated server")
        if any(e["fields"].get("sampleKind") == "afterRelease" for e in samples):
            return events, request_ids.pop()
        time.sleep(0.2)
    raise RuntimeError("No afterRelease sample: request still active or telemetry unavailable")


def stream_request(url, payload, cancel):
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-session-affinity": "capture-memory-replay"},
    )
    chunks = []
    usage = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                return chunks, usage, False
            chunk = json.loads(line[6:])
            if "error" in chunk:
                raise RuntimeError("Server returned a streaming error")
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                chunks.append(delta)
                if cancel and any(delta.get(key) for key in ("content", "reasoning_content", "tool_calls")):
                    return chunks, usage, True
    raise RuntimeError("Stream ended without [DONE]")


def assistant_message(chunks):
    message = {"role": "assistant", "content": ""}
    calls = {}
    for chunk in chunks:
        for key in ("content", "reasoning_content"):
            if chunk.get(key):
                message[key] = message.get(key, "") + chunk[key]
        for call in chunk.get("tool_calls", []):
            target = calls.setdefault(call["index"], {"type": "function", "function": {}})
            if call.get("id"):
                target["id"] = call["id"]
            for key, value in call.get("function", {}).items():
                target["function"][key] = target["function"].get(key, "") + value
    if calls:
        message["tool_calls"] = [calls[key] for key in sorted(calls)]
    return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--expect-capture-mode", choices=("copy", "handoff"))
    parser.add_argument("--base-url", default="http://127.0.0.1:18321")
    parser.add_argument("--diagnostics", type=pathlib.Path, default=pathlib.Path.home()
                        / "Library/Application Support/CacheDiagnostics"
                        / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d") + ".jsonl"))
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--cancel-after-first-delta", action="store_true")
    parser.add_argument("--next-request", type=pathlib.Path)
    parser.add_argument("--release-timeout", type=float, default=300)
    args = parser.parse_args()
    endpoint = urllib.parse.urlparse(args.base_url)
    if endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "localhost", "::1"):
        parser.error("Replay accepts only a loopback HTTP server")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.cancel_after_first_delta and args.next_request:
        parser.error("A cancelled turn cannot create a continuation fixture")
    if args.output.exists() or (args.next_request and args.next_request.exists()):
        parser.error("Refusing to overwrite an existing evidence file or continuation")

    payload = read_request(args.request)
    payload.pop("max_tokens", None)
    payload.update(model=args.model, max_completion_tokens=args.max_tokens,
                   temperature=0, stream=True, stream_options={"include_usage": True})
    stat = args.diagnostics.stat()
    offset, identity = stat.st_size, (stat.st_dev, stat.st_ino)
    started = time.monotonic()
    chunks, usage, cancelled = stream_request(args.base_url, payload, args.cancel_after_first_delta)
    response_seconds = time.monotonic() - started
    events, request_id = await_release(args.diagnostics, offset, identity, args.release_timeout)
    message = assistant_message(chunks)
    evidence = {
        "label": args.label, "sourceRevision": args.source_revision,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "requestID": request_id, "requestSHA256": digest(payload),
        "responseSHA256": digest(message), "model": args.model,
        "maxCompletionTokens": args.max_tokens, "temperature": 0,
        "reasoningEffort": payload.get("reasoning_effort"),
        "cancelled": cancelled, "responseSeconds": response_seconds,
        "releaseObservedSeconds": time.monotonic() - started,
        "usage": usage, "events": [e for e in events if e.get("requestID") == request_id],
        # Writer events have system scope. They may belong to an older
        # request, so retain their snapshot IDs without attributing them to
        # the request whose afterRelease sample ended this observation window.
        "systemEventsObserved": [e for e in events if not e.get("requestID")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    if args.expect_capture_mode and not cancelled:
        modes = {e["fields"]["leafCaptureMode"] for e in events
                 if e["eventName"] == "requestMemory" and "leafCaptureMode" in e["fields"]}
        if modes != {args.expect_capture_mode}:
            raise RuntimeError(f"Expected capture mode {args.expect_capture_mode}, observed {sorted(modes)}")
    if args.next_request:
        payload["messages"].append(message)
        if message.get("tool_calls"):
            for call in message["tool_calls"]:
                payload["messages"].append({"role": "tool", "tool_call_id": call["id"],
                                            "content": "Tool execution omitted for this cache replay."})
        else:
            payload["messages"].append({"role": "user", "content": "Continue."})
        # These contain private corpus text: use restrictive file permissions.
        with open(args.next_request, "x", opener=lambda p, flags: os.open(p, flags, 0o600)) as target:
            json.dump(payload, target)
    print(json.dumps({key: value for key, value in evidence.items() if key not in {"events"}}))


if __name__ == "__main__":
    main()
