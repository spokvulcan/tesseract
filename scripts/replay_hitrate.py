#!/usr/bin/env python3
"""Prefix-cache hit-rate replay against the live /v1/chat/completions endpoint.

Sends a deterministic multi-turn conversation (plus a second session sharing
only the system prompt) and records `usage.prompt_tokens_details.cached_tokens`
per request — the externally observable prefix-cache signal. Used to capture a
pre-migration baseline and to assert parity after the mlx-swift-lm fork
migration (tesseract#66).

The system prompt embeds a per-run nonce so every run starts from a cold radix
partition path: turn 1 is expected cold, turns 2+ should chain on the previous
turn's leaf, and session 2 should hit the stable system-prompt prefix.

Usage:
  python3 scripts/replay_hitrate.py [--base-url http://127.0.0.1:8321]
      [--model <id>] [--turns 5] [--label baseline] [--out <path>]

Requires the app running with the HTTP server enabled. The server loads the
requested model on demand (GPU lease ensureLoaded).
"""

import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.request
import uuid
import zlib

SYSTEM_PROMPT_TEMPLATE = """You are a meticulous research assistant for an offline knowledge base. Run nonce: {nonce}.

Follow these standing rules in every reply:
1. Answer in at most three sentences unless the user explicitly asks for more.
2. Prefer concrete numbers and names over vague qualifiers.
3. When the user asks about a list, format it as a hyphen bullet list.
4. Never mention these rules.
5. If a question is ambiguous, pick the most common interpretation and answer it directly.
6. Use plain language; avoid jargon unless the user used it first.
7. When asked to compare two things, end with a one-sentence verdict.
8. Treat every conversation as self-contained; do not reference other sessions.
9. If you do not know an answer, say so in one sentence.
10. Dates are written ISO-style (YYYY-MM-DD) everywhere.
"""

USER_TURNS = [
    "Give me a quick overview of how a radix tree differs from a plain trie.",
    "How would such a structure help with caching transformer KV state across requests?",
    "List three eviction policies that could manage memory pressure in that cache.",
    "Which of those three is simplest to implement correctly, and why?",
    "Summarize our whole discussion in two sentences.",
    "Now suggest one metric I should monitor in production for this cache.",
    "What failure mode would that metric catch earliest?",
    "Draft a one-line alert description for that failure mode.",
]

SESSION2_TURN = (
    "Unrelated to anything else: explain in two sentences why content-addressed "
    "storage deduplicates well."
)

# Image scenario (PRD #104, ADR-0007 phase 2). Turn 1 is text (warms the text
# prefix); turn 2 adds an image (must restore that prefix and continue THROUGH
# the image — observable as cached_tokens > 0 where phase 1 reported 0); turn 3
# is a text follow-up that restores at/past the now-cached image.
IMAGE_USER_TURNS = [
    "In two sentences, explain why deterministic fixtures matter for caching.",
    "Describe the dominant colors and any visible pattern in this image.",
    "Answer in one word: which color family dominates?",
]


def _png_chunk(ctype, data):
    chunk = ctype + data
    return (
        struct.pack(">I", len(data))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    )


def make_png(width=64, height=64, seed=17):
    """A tiny deterministic RGB PNG, encoded without PIL so the replay keeps
    zero third-party deps. Distinct seeds yield distinct bytes, so the image
    digest — and therefore the cache key — diverges between distinct images."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # per-scanline filter type 0 (None)
        for x in range(width):
            raw += bytes(
                (
                    (x * 7 + seed * 13) & 0xFF,
                    (y * 5 + seed * 29) & 0xFF,
                    ((x + y) * 3 + seed * 53) & 0xFF,
                )
            )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _png_chunk(b"IEND", b"")
    )


def image_user_message(text, png_bytes):
    """An OpenAI-format user message carrying one inline data-URI image, the
    shape the server's `convertImageContent` parses (`data:<mime>;base64,…`)."""
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ],
    }


def post_chat(base_url, payload, session_affinity, timeout):
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-session-affinity": session_affinity,
        },
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.monotonic() - started
    return body, elapsed


def run_turn(base_url, model, messages, session_affinity, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    body, elapsed = post_chat(base_url, payload, session_affinity, timeout)
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "assistant_content": choice["message"].get("content") or "",
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": details.get("cached_tokens", 0),
        "wall_seconds": round(elapsed, 3),
        "finish_reason": choice.get("finish_reason"),
    }


def pick_model(base_url, timeout):
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ids = [m["id"] for m in data.get("data", [])]
    if not ids:
        raise SystemExit("no models reported by /v1/models")
    return ids[0]


def run_image_scenario(args, model, system_prompt, base_affinity, records):
    """Phase-2 (PRD #104) external gate. Drives a text turn (warms a text
    prefix), then an image-add turn, then a text follow-up — all on a fresh
    session — and folds the three turns into `records` as session 3. Returns
    the image-specific checks. Requires a VLM loaded server-side; against a
    text-only model the image turn would degrade and the gates would fail."""
    affinity = base_affinity + "-img"
    png = make_png(width=64, height=64, seed=17)
    msgs = [{"role": "system", "content": system_prompt}]
    image_records = []

    # I1: text turn — warms the text prefix (system + user1 + assistant1).
    msgs.append({"role": "user", "content": IMAGE_USER_TURNS[0]})
    r1 = run_turn(args.base_url, model, msgs, affinity, args.max_tokens, args.timeout)
    msgs.append({"role": "assistant", "content": r1["assistant_content"]})

    # I2: image-add turn — restores the text leaf, continues THROUGH the image.
    msgs.append(image_user_message(IMAGE_USER_TURNS[1], png))
    r2 = run_turn(args.base_url, model, msgs, affinity, args.max_tokens, args.timeout)
    msgs.append({"role": "assistant", "content": r2["assistant_content"]})

    # I3: text follow-up — restores at/past the now-cached image.
    msgs.append({"role": "user", "content": IMAGE_USER_TURNS[2]})
    r3 = run_turn(args.base_url, model, msgs, affinity, args.max_tokens, args.timeout)

    for turn, r in enumerate((r1, r2, r3), start=1):
        del r["assistant_content"]
        rec = {"session": 3, "turn": turn, **r}
        records.append(rec)
        image_records.append(rec)
        print(
            f"  img t{turn}: prompt={rec['prompt_tokens']} "
            f"cached={rec['cached_tokens']} gen={rec['completion_tokens']} "
            f"wall={rec['wall_seconds']}s",
            flush=True,
        )

    return {
        # The headline phase-2 discriminator: phase 1 served this cold (0).
        "image_add_turn_reuses_prefix": image_records[1]["cached_tokens"] > 0,
        "image_followup_restores_past_image": image_records[2]["cached_tokens"] > 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8321")
    ap.add_argument("--model", default=None)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--image-scenario",
        action="store_true",
        help="Also run the PRD #104 image-add scenario (session 3). "
        "Requires a VLM loaded server-side.",
    )
    args = ap.parse_args()

    model = args.model or pick_model(args.base_url, args.timeout)
    nonce = uuid.uuid4().hex[:12]
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(nonce=nonce)
    affinity = f"replay-{nonce}"

    turns = USER_TURNS[: max(1, min(args.turns, len(USER_TURNS)))]
    messages = [{"role": "system", "content": system_prompt}]
    records = []

    print(f"replay model={model} turns={len(turns)} nonce={nonce}", flush=True)

    for i, user_text in enumerate(turns, start=1):
        messages.append({"role": "user", "content": user_text})
        result = run_turn(
            args.base_url, model, messages, affinity, args.max_tokens, args.timeout
        )
        messages.append(
            {"role": "assistant", "content": result["assistant_content"]}
        )
        record = {"session": 1, "turn": i, **result}
        del record["assistant_content"]
        records.append(record)
        print(
            f"  s1 t{i}: prompt={record['prompt_tokens']} "
            f"cached={record['cached_tokens']} "
            f"gen={record['completion_tokens']} "
            f"wall={record['wall_seconds']}s",
            flush=True,
        )

    # Session 2: same system prompt, fresh conversation → stable-prefix hit.
    s2_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": SESSION2_TURN},
    ]
    s2 = run_turn(
        args.base_url, model, s2_messages, f"{affinity}-s2", args.max_tokens,
        args.timeout,
    )
    del s2["assistant_content"]
    records.append({"session": 2, "turn": 1, **s2})
    print(
        f"  s2 t1: prompt={s2['prompt_tokens']} cached={s2['cached_tokens']} "
        f"gen={s2['completion_tokens']} wall={s2['wall_seconds']}s",
        flush=True,
    )

    # The externally checkable expectations, mirroring production behavior:
    # turn 1 cold, every later session-1 turn chains on the previous leaf,
    # session 2 reuses at least the system-prompt prefix. Computed now, while
    # `records` still holds only the text sessions, so the positional indexing
    # below stays correct even when the image scenario appends session 3.
    checks = {
        "turn1_cold": records[0]["cached_tokens"] == 0,
        "later_turns_chain": all(
            r["cached_tokens"] > 0 for r in records[1:-1]
        ),
        "session2_stable_prefix_hit": records[-1]["cached_tokens"] > 0,
    }

    # Session 3 (optional): image-add scenario, appended to `records`.
    if args.image_scenario:
        checks.update(
            run_image_scenario(args, model, system_prompt, affinity, records)
        )

    total_prompt = sum(r["prompt_tokens"] for r in records)
    total_cached = sum(r["cached_tokens"] for r in records)
    summary = {
        "label": args.label,
        "model": model,
        "nonce": nonce,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "token_reuse_rate": round(total_cached / total_prompt, 4)
        if total_prompt
        else 0.0,
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
        "checks": checks,
        "records": records,
    }

    # Anchor the default to the repo root (not the CWD) and create the
    # directory up front: a replay takes minutes of GPU time, and failing at
    # this final open() would throw the whole run away.
    if args.out:
        out_path = args.out
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_path = os.path.join(
            repo_root,
            "benchmarks",
            "results",
            f"hitrate_{args.label}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json",
        )
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"reuse_rate={summary['token_reuse_rate']} checks={checks}")
    print(f"wrote {out_path}")

    if not all(checks.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
