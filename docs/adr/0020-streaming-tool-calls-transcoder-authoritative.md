---
status: accepted
---

# Streaming tool calls: transcoder-authoritative Argument Fragments

The server streams tool-call arguments incrementally on `/v1/chat/completions`
(OpenAI `delta.tool_calls[].function.arguments` fragments) so clients like
OpenCode render a file edit or write as it is generated, instead of a frozen
screen until the call completes. The in-flight deltas already exist internally
(`.toolCallDelta` / `toolCallBufferDelta`, ADR-0006); what ships here is the
**Argument Transcoder** — per-format conversion of model-native tool-call text
(Qwen3.5/3.6 `<function=…>` XML and the `<tool_call>` JSON wrapper body) into
wire fragments — always on for every streaming client, with the atomic
name-then-full-arguments emission remaining only for formats the transcoder
does not understand.

Three decisions a future reader will otherwise "fix":

1. **The streamed fragments are authoritative for the wire; the final parse is
   only a diagnostic.** Once fragments are sent, the final chunk carries only
   `finish_reason: tool_calls` — the full arguments are never re-sent (the
   client concatenates *all* fragments; a trailing full copy corrupts the JSON).
   Byte-exact agreement with a re-serialization of the parsed arguments dict is
   impossible anyway (dictionary key order is nondeterministic), so the server
   compares transcoder output against the parser's final tool call
   *semantically* and logs a mismatch diagnostic — it never retro-corrects.
2. **Wire-Valid Close instead of the malformed→text fallback.** There is no
   retraction on the OpenAI wire: after a tool call's id and name are streamed,
   a malformation, cancel, intervention, or max-tokens termination synthesizes
   closers so the accumulated arguments still parse, and the client's normal
   tool-error loop handles recovery. The malformed→text fallback survives only
   where nothing was streamed yet (non-streaming path, or malformation before
   name-lock — the transcoder engages late, at name-lock, to keep that window
   wide).
3. **No parseable strict prefix.** The AI SDK (OpenCode's parser) finalizes a
   tool call the moment the accumulated arguments parse as JSON, so the
   transcoder never closes the object mid-call: string parameter values stream
   progressively (JSON-escaped), schema-typed non-string values (the vendor
   XML parser types values from the request's tool schema) are buffered and
   emitted whole at parameter close, and `}` is emitted exactly once at
   function close.

One deviation from the plan's "the delta events already carry everything
needed": the app-owned delta tracker withheld a close-tag chunk's pre-close
body bytes (they arrived inside the same chunk as `</tool_call>` and were
swallowed with the tag), so the last argument bytes would have vanished from
the wire. The tracker now deltas those body bytes — the tag itself is still
never a delta — which is a strict widening of what downstream consumers see,
not a format change.

Rejected: parser-authoritative streaming (re-send authoritative args at the
end — corrupts client concatenation); a config gate for the fragment format
(it is the canonical OpenAI streaming shape; two emission shapes double the
test surface for no known client); holding fragments until the first parameter
closes cleanly (narrows the malformation window but forfeits progressive
display of the first — often the only large — parameter).
