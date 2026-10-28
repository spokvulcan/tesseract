"""Tool-prompt renderings of the app's chat templates, as transformers renders them.

The reference ToolPromptParityTests compares the app's renders against. The
tools mirror what AgentToolDefinition.toolSpec builds, in the order the app's
sources write them; the test builds the same tools through that builder.

Usage: python3 generate.py OUT_JSON NAME=MODEL_DIR [NAME=MODEL_DIR ...]
Each model dir needs tokenizer.json, tokenizer_config.json and
chat_template.jinja. NAME is the key the test reads (qwen3.5, qwen3.8, lfm3,
lfm3-audio).
"""
import hashlib
import json
import re
import sys

import transformers
from transformers import PreTrainedTokenizerFast


def prop(type_, description, enum=None, items=None):
    d = {"type": type_, "description": description}
    if enum is not None:
        d["enum"] = enum
    if items is not None:
        d["items"] = items
    return d


def tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS = [
    tool("get_weather", "Weather for a city.", {"city": prop("string", "The city's name.")}, ["city"]),
    tool(
        "read",
        "Read the contents of a file. Returns numbered lines (cat -n style).",
        {
            "path": prop("string", "File path relative to working directory"),
            "offset": prop("integer", "Line number to start from (1-indexed)"),
            "limit": prop("integer", "Maximum number of lines to return"),
        },
        ["path"],
    ),
    tool(
        "edit",
        "Edit a file by replacing an exact text match.",
        {
            "path": prop("string", "File path to edit"),
            "old_text": prop("string", "Exact text to find and replace"),
            "new_text": prop("string", "Replacement text"),
        },
        ["path", "old_text", "new_text"],
    ),
    tool(
        "search_notes",
        "Search notes — by title, tag or text such as </style> or \"quoted words\".",
        {
            "query": prop("string", "Text to find, e.g. café"),
            "tags": prop("array", "Tags to match", items=prop("string", "A tag", enum=["work", "home"])),
            "sort": prop("string", "Result order", enum=["newest", "oldest"]),
            "limit": prop("integer", "At most this many notes"),
            "archived": prop("boolean", "Include archived notes"),
        },
        ["query"],
    ),
]

SYSTEM = "You are a careful assistant."

CASES = {
    "tools": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "What's the weather in Paris?"},
    ],
    # Argument keys in sorted order: the app holds parsed arguments in a Swift
    # dictionary, so their order is not part of this check.
    "tool_call_history": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Find my newest café notes tagged work or home, five at most."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_notes",
                        "arguments": {
                            "limit": 5,
                            "query": "café",
                            "sort": "newest",
                            "tags": ["work", "home"],
                        },
                    },
                }
            ],
        },
        {"role": "tool", "content": "2 notes: Lyon café, Café crawl"},
    ],
    "nested_argument": [
        {"role": "user", "content": "Note the café filter."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_notes",
                        "arguments": {"query": {"after": "2026-01-01", "text": "café"}},
                    },
                }
            ],
        },
        {"role": "tool", "content": "ok"},
    ],
}

out_path = sys.argv[1]
out = {"transformers": transformers.__version__, "tools": TOOLS, "cases": CASES, "renders": {}}
for spec in sys.argv[2:]:
    name, directory = spec.split("=", 1)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(directory)
    template = open(f"{directory}/chat_template.jinja").read()
    renders = {}
    for case, messages in CASES.items():
        text = tokenizer.apply_chat_template(
            messages, tools=TOOLS, add_generation_prompt=True, tokenize=False, chat_template=template)
        renders[case] = {"text": text, "ids": tokenizer.encode(text, add_special_tokens=False)}
    out["renders"][name] = {
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "cases": renders,
    }
    print(name, {c: len(r["ids"]) for c, r in renders.items()})
text = json.dumps(out, indent=1, ensure_ascii=False)
# One line per id array, so a regenerated fixture diffs by case.
text = re.sub(r"\[\s+(-?\d+(?:,\s+-?\d+)*)\s+\]", lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", text)
open(out_path, "w").write(text + "\n")
