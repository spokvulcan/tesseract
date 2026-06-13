#!/usr/bin/env bash
set -euo pipefail

# Interrupt-readiness live drill (PRD #94, issue #101).
#
# Reproduces the 2026-06-12 incident shape against a running Tesseract
# server and measures post-interrupt time-to-first-token (TTFT). The
# incident recorded 92.8 s; the acceptance bar is <= 5 s.
#
#   Stretch Abandonment drill (default):
#     1. Send a tool-stretch request, then ABORT it mid-generation.
#     2. Wait past the Stretch Abandonment idle window (5 s) so the
#        speculative canonical pass has GPU time to build the spine.
#     3. Send the steering message; measure its TTFT.
#
#   Double-interrupt drill (--double):
#     1..3 as above, but ALSO abort the recovery prefill shortly after it
#        starts, then re-send. The retry must resume from the salvaged
#        partial (salvage-on-cancel, #97), never restart from the floor —
#        observable as the second retry's TTFT being far below a cold
#        re-prefill of the whole stretch.
#
# The server must be running with the prefix cache enabled and the
# incident model loaded. Start it from the app (Settings → Server) or a
# dev-release build, then:
#
#   scripts/interrupt-drill.sh \
#     --port 8321 \
#     --model qwen3.5-4b-paro \
#     --prompt ~/projects/tesseract-traces/2026-06-12-interrupt-rewind/drill-stretch.json
#
# --prompt is an OpenAI /v1/chat/completions request body whose last
# assistant turn is a long-running tool call (the "stretch"); the steering
# body is derived from it by appending the tool result + a user message,
# or supplied via --steer. Both live in the incident corpus, kept out of
# the repo.

PORT=8321
MODEL="qwen3.5-4b-paro"
PROMPT=""
STEER=""
IDLE_WINDOW=6          # seconds; one second past the 5 s abandonment window
ABORT_AFTER=2          # seconds into a request before aborting it
DOUBLE=0
SESSION="drill-$(date +%s)"

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --steer) STEER="$2"; shift 2 ;;
    --idle-window) IDLE_WINDOW="$2"; shift 2 ;;
    --abort-after) ABORT_AFTER="$2"; shift 2 ;;
    --double) DOUBLE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$PROMPT" ] || [ ! -f "$PROMPT" ]; then
  echo "error: --prompt <request.json> is required and must exist" >&2
  exit 2
fi
if [ -n "$STEER" ] && [ ! -f "$STEER" ]; then
  echo "error: --steer <request.json> must exist" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required to stamp model and derive steering requests" >&2
  exit 2
fi

TEMP_FILES=()
cleanup() {
  for file in "${TEMP_FILES[@]}"; do
    rm -f "$file"
  done
}
trap cleanup EXIT

BASE="http://127.0.0.1:${PORT}/v1/chat/completions"
HDR_SESSION="x-session-affinity: ${SESSION}"

with_model() {
  local source="$1"
  local target
  target="$(mktemp -t interrupt-drill-body).json"
  TEMP_FILES+=("$target")
  jq --arg model "$MODEL" '.model = $model' "$source" > "$target"
  echo "$target"
}

PROMPT_BODY="$(with_model "$PROMPT")"

# Derive the steering body if not supplied: append a tool result + user
# steer to the stretch request's messages.
if [ -z "$STEER" ]; then
  STEER_SOURCE="$(mktemp -t interrupt-drill-steer).json"
  TEMP_FILES+=("$STEER_SOURCE")
  jq '
      (.messages
        | map(select(.role == "assistant" and ((.tool_calls // []) | length > 0)))
        | last
        | .tool_calls[0].id) as $toolCallID
      | .messages += [
        {
          "role":"tool",
          "tool_call_id":($toolCallID // "drill-tool-call"),
          "content":"(tool result arrived after the interrupt)"
        },
        {"role":"user","content":"Stop — summarize what you have and move on."}
      ]' "$PROMPT_BODY" > "$STEER_SOURCE"
else
  STEER_SOURCE="$STEER"
fi
STEER_BODY="$(with_model "$STEER_SOURCE")"

# Fire a request in the background and abort it after ABORT_AFTER seconds.
fire_and_abort() {
  local body="$1" tag="$2"
  curl -sS -N -X POST "$BASE" \
    -H 'content-type: application/json' -H "$HDR_SESSION" \
    --data @"$body" >/dev/null 2>&1 &
  local pid=$!
  sleep "$ABORT_AFTER"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  echo "[drill] ${tag}: aborted after ${ABORT_AFTER}s" >&2
}

# Measure TTFT of a streamed request: seconds until the first SSE data
# chunk. Uses curl's first-byte timing on the streamed body.
measure_ttft() {
  local body="$1"
  curl -sS -N -X POST "$BASE" \
    -H 'content-type: application/json' -H "$HDR_SESSION" \
    -o /dev/null \
    -w '%{time_starttransfer}\n' \
    --data @"$body"
}

echo "[drill] session=${SESSION} model=${MODEL} double=${DOUBLE}"
echo "[drill] 1. tool stretch → abort mid-generation"
fire_and_abort "$PROMPT_BODY" "stretch"

echo "[drill] 2. idle ${IDLE_WINDOW}s (> abandonment window) — speculative spine builds"
sleep "$IDLE_WINDOW"

if [ "$DOUBLE" -eq 1 ]; then
  echo "[drill] 3a. steering message → abort the recovery prefill mid-flight"
  fire_and_abort "$STEER_BODY" "recovery"
  echo "[drill] 3b. re-send steering — must resume from the salvage, not the floor"
fi

echo "[drill] 4. steering message → measuring post-interrupt TTFT"
TTFT="$(measure_ttft "$STEER_BODY")"
echo "[drill] post-interrupt TTFT: ${TTFT}s (incident: 92.8s · bar: <= 5s)"

# Exit non-zero if the bar is missed, so CI / a wrapper can gate on it.
awk -v t="$TTFT" 'BEGIN { exit (t+0 <= 5.0) ? 0 : 1 }' || {
  echo "[drill] FAIL: TTFT ${TTFT}s exceeds the 5 s bar" >&2
  exit 1
}
echo "[drill] PASS: TTFT within the 5 s bar"
