"""Does the rendering of the tools block change tool calling?

Renders the same 13 app tools five ways, prefills each tools block once, then
for every task: the teacher-forced log-probability of the gold function name,
and a greedy decode checked for the right function, parameter names and values
(order-insensitive). Usage: probe_tool_order.py MODEL_DIR OUT_JSON
Ran with mlx-lm 0.32.2, mlx 0.32.4, transformers 5.21.0 (mlx-lm 0.30.8 loads
mlx-community/Qwen3.8-27B-4bit wrongly and generates noise).
"""
import json
import os
import re
import sys
import time

import mlx.core as mx
from mlx_lm import load
from transformers.utils import chat_template_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app_tools import APP_TOOLS, rank_keys, sort_keys  # noqa: E402

model_dir, out_path = sys.argv[1:3]
SYSTEM = "You are a helpful assistant on the owner's Mac. Today is Wednesday 2026-10-28, 14:05."

# (user message, function, {param: value}) — values compared as strings.
TASKS = [
    ("Read notes/todo.md", "read", {"path": "notes/todo.md"}),
    ("Show me lines 40 to 60 of Sources/App/main.swift", "read", {"path": "Sources/App/main.swift", "offset": "40", "limit": "21"}),
    ("What's in the docs folder?", "ls", {"path": "docs"}),
    ("List the first 20 entries of the current directory", "ls", {"limit": "20"}),
    ("Create hello.txt containing exactly: Hello, world!", "write", {"path": "hello.txt", "content": "Hello, world!"}),
    ("Overwrite config/flags.txt so it contains only the line: beta=on", "write", {"path": "config/flags.txt", "content": "beta=on", "overwrite": "true"}),
    ("In README.md, change the text 'Version 1.2' to 'Version 1.3'", "edit", {"path": "README.md", "old_text": "Version 1.2", "new_text": "Version 1.3"}),
    ("In src/app.py replace the line DEBUG = True with DEBUG = False", "edit", {"path": "src/app.py", "old_text": "DEBUG = True", "new_text": "DEBUG = False"}),
    ("Remember that I'm allergic to peanuts", "remember", {}),
    ("Remember that I work from home on Fridays, under Work", "remember", {"area": "Work"}),
    ("Forget the fact that I live in Berlin", "forget", {}),
    ("What did I tell you about my dentist?", "recall", {}),
    ("What did I say about the trip in the last 7 days?", "recall", {"days": "7"}),
    ("What's on my calendar today?", "agenda", {}),
    ("What do I have tomorrow?", "agenda", {"from": "tomorrow"}),
    ("Show my agenda for the next 5 days", "agenda", {"days": "5"}),
    ("Remind me to buy milk", "add_reminder", {}),
    ("Remind me to call mom today at 18:00", "add_reminder", {"due": "2026-10-28T18:00"}),
    ("Add a reminder in my Health list to book a physio appointment", "add_reminder", {"area": "Health"}),
    ("Mark reminder R-17 as done", "update_reminder", {"id": "R-17", "completed": "true"}),
    ("Rename reminder R-9 to Pay the electricity bill", "update_reminder", {"id": "R-9", "title": "Pay the electricity bill"}),
    ("Move reminder R-4 to Friday 2026-10-30", "update_reminder", {"id": "R-4", "due": "2026-10-30"}),
    ("Block 2026-10-29 from 09:00 to 11:00 for Deep work", "add_event", {"title": "Deep work", "start": "2026-10-29T09:00", "end": "2026-10-29T11:00"}),
    ("Add lunch with Sam tomorrow at 12:30 for 60 minutes", "add_event", {"start": "2026-10-29T12:30", "duration_minutes": "60"}),
    ("Put a dentist appointment on 2026-11-03 at 15:00 at Main Street Clinic", "add_event", {"start": "2026-11-03T15:00", "location": "Main Street Clinic"}),
    ("Move event E-12 to 2026-10-29T16:00", "move_event", {"id": "E-12", "start": "2026-10-29T16:00"}),
    ("Move event E-3 to start 2026-10-30T10:00 and end 2026-10-30T10:30", "move_event", {"id": "E-3", "start": "2026-10-30T10:00", "end": "2026-10-30T10:30"}),
    ("Delete event E-21", "delete_event", {"id": "E-21"}),
    ("Cancel my focus block E-8, I don't need it", "delete_event", {"id": "E-8"}),
    ("Open the file Package.swift and show it to me", "read", {"path": "Package.swift"}),
]


class Compact:
    """`json` shim: what the app's tojson writes today (sorted keys, no spaces)."""

    @staticmethod
    def dumps(x, **kwargs):
        kwargs.update(separators=(",", ":"), sort_keys=True)
        return json.dumps(x, **kwargs)


class CompactSourceOrder:
    @staticmethod
    def dumps(x, **kwargs):
        kwargs.update(separators=(",", ":"))
        return json.dumps(x, **kwargs)


VARIANTS = {
    "python": (APP_TOOLS, json),
    "swift_today": (APP_TOOLS, Compact),
    "sorted_spaced": ([sort_keys(t) for t in APP_TOOLS], json),
    "ranked_spaced": ([rank_keys(t) for t in APP_TOOLS], json),
    "python_compact": (APP_TOOLS, CompactSourceOrder),
}

model, tokenizer = load(model_dir)
template = open(f"{model_dir}/chat_template.jinja").read()


def render(tools, json_module, user):
    saved = chat_template_utils.json
    chat_template_utils.json = json_module
    try:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tools=tools, add_generation_prompt=True, tokenize=False,
            chat_template=template, enable_thinking=False)
    finally:
        chat_template_utils.json = saved


def encode(text):
    return tokenizer.encode(text, add_special_tokens=False)


def snapshot(cache):
    return [c.state for c in cache]


def restore(states):
    cache = model.make_cache()
    for c, s in zip(cache, states):
        c.state = list(s) if isinstance(s, list) else s
    return cache


def gold_call(fn, args_text):
    return f"<tool_call>\n<function={fn}>\n" + args_text


def parse_call(text):
    m = re.search(r"<function=([^>\n]+)>\n(.*?)</function>", text, re.S)
    if not m:
        return None, {}
    params = dict(re.findall(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", m.group(2), re.S))
    return m.group(1), params


results = {}
for name, (tools, json_module) in VARIANTS.items():
    t0 = time.time()
    probe = render(tools, json_module, "PLACEHOLDER")
    prefix_text = probe[: probe.index("<|im_start|>user\n")]
    prefix = encode(prefix_text)
    cache = model.make_cache()
    logits = model(mx.array([prefix]), cache=cache)
    mx.eval(logits, [c.state for c in cache])
    states = snapshot(cache)
    rows = []
    for user, fn, expected in TASKS:
        full = encode(render(tools, json_module, user))
        assert full[: len(prefix)] == prefix, name
        suffix = full[len(prefix):]
        # Teacher-forced log-prob of "<tool_call>\n<function=NAME>\n".
        gold = encode(gold_call(fn, ""))
        cache = restore(states)
        out = model(mx.array([suffix + gold]), cache=cache)
        logp = out[0].astype(mx.float32)
        logp = logp - mx.logsumexp(logp, axis=-1, keepdims=True)
        start = len(suffix) - 1
        positions = mx.arange(start, start + len(gold))
        token_logp = logp[positions, mx.array(gold)]
        name_logp = float(mx.sum(token_logp).item())
        # Greedy decode from the generation prompt.
        cache = restore(states)
        step = model(mx.array([suffix]), cache=cache)
        generated = []
        for _ in range(160):
            token = int(mx.argmax(step[0, -1]).item())
            generated.append(token)
            if token in tokenizer.eos_token_ids:
                break
            text = tokenizer.decode(generated)
            if "</tool_call>" in text or "<|im_end|>" in text:
                break
            step = model(mx.array([[token]]), cache=cache)
        text = tokenizer.decode(generated)
        got_fn, got = parse_call(text)
        values_ok = all(got.get(k, "").strip() == v for k, v in expected.items())
        rows.append({
            "user": user, "function": fn, "name_logp": name_logp,
            "fn_ok": got_fn == fn, "values_ok": got_fn == fn and values_ok,
            "param_order": list(got), "output": text,
        })
    results[name] = {"prefix_tokens": len(prefix), "rows": rows, "seconds": time.time() - t0}
    fn_acc = sum(r["fn_ok"] for r in rows) / len(rows)
    val_acc = sum(r["values_ok"] for r in rows) / len(rows)
    mean_logp = sum(r["name_logp"] for r in rows) / len(rows)
    print(f"{name:15s} prefix={len(prefix):5d} fn_acc={fn_acc:.3f} values_acc={val_acc:.3f} "
          f"mean_name_logp={mean_logp:.4f} ({time.time() - t0:.0f}s)", flush=True)

json.dump(results, open(out_path, "w"), indent=1)
