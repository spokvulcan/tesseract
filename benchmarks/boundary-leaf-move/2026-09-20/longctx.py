#!/usr/bin/env python3
"""Bounded long-context agent-shaped replay against the local Tesseract server.

Grows one conversation to ~30k tokens over N stop-finish turns. Every
stop-finish turn under a think-stripping template takes the Leaf Store's
boundary path, which is the path under measurement. Watchdog aborts if the
process footprint crosses the stop threshold.
"""
import json, subprocess, sys, time, urllib.request, pathlib

URL = "http://127.0.0.1:8321/v1/chat/completions"
MODEL = "qwen3.8-27b"
STOP_GB = 38.0
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
PAD_CHARS = int(sys.argv[2]) if len(sys.argv) > 2 else 18000

pad_source = pathlib.Path("/Users/owl/projects/tesseract/CONTEXT.md").read_text()

def footprint_gb():
    try:
        out = subprocess.run(["ps", "-A", "-o", "rss=,comm="], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "Tesseract Agent" in line:
                return int(line.split()[0]) / 1024 / 1024
    except Exception:
        pass
    return 0.0

def post(messages):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": 320,
        "reasoning_effort": "low",
        "chat_template_kwargs": {"preserve_thinking": False},
        "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        payload = json.load(r)
    return payload, time.time() - t0

messages = [{"role": "system", "content": "You are a terse coding assistant. Answer in one short sentence."}]
offset = 0
for i in range(ROUNDS):
    rss = footprint_gb()
    if rss > STOP_GB:
        print(f"STOP: footprint {rss:.1f} GB over threshold {STOP_GB} GB", flush=True)
        break
    chunk = pad_source[offset:offset + PAD_CHARS]
    offset += PAD_CHARS
    messages.append({
        "role": "user",
        "content": f"Here is part {i+1} of a document:\n\n{chunk}\n\nIn one sentence, what is this part about?",
    })
    payload, secs = post(messages)
    usage = payload.get("usage", {})
    msg = payload["choices"][0]["message"]
    assistant = {"role": "assistant", "content": msg.get("content") or ""}
    # Deliberately NOT echoing reasoning_content: a client that drops it
    # puts the next request under the think-stripping render, which is the
    # Leaf Store's boundary path — the path under measurement.
    messages.append(assistant)
    print(f"round {i+1}: prompt_tokens={usage.get('prompt_tokens')} "
          f"completion={usage.get('completion_tokens')} {secs:.1f}s rss={footprint_gb():.1f}GB", flush=True)
print("done", flush=True)
