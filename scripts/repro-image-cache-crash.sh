#!/usr/bin/env bash
# Repro harness for the image prefix-cache "Invalid Resource" crash.
#
# ROOT CAUSE (confirmed): an MLX-core use-after-free, NOT app code. Command
# buffers are created with commandBufferWithUnretainedReferences(), so Metal
# does not keep referenced MTLBuffers alive. Under the memory pressure of warm
# prefix-cache restore + re-prefill on a 27B vision model, the allocator's
# buffer-cache trim (malloc -> release_cached_buffers -> buf->release) frees an
# MTLBuffer that an in-flight command buffer still references ->
# kIOGPUCommandBufferCallbackErrorInvalidResource -> SIGABRT on
# com.Metal.CompletionQueueDispatch. Timing-sensitive: any slowdown hides it.
#
# FIX: retained-reference command buffers in the MLX Metal backend.
#   Upstream PR:    https://github.com/ml-explore/mlx/pull/3688
#   Upstream issue: https://github.com/ml-explore/mlx/issues/3689
# We currently consume this via a fork pin (Vendor/mlx-{swift-lm,audio-swift}/
# Package.swift -> spokvulcan/mlx-swift). Kept for reproducibility/discussion
# while the upstream PR is in review; remove once it ships (see the TODO(mlx-uaf)
# markers in those Package.swift files).
#
# This replays a REAL recorded image request against the LIVE app server in a
# tight loop and watches for the app to die. Fully automated: no GUI clicking.
# To VERIFY THE CRASH still exists (e.g. on upstream/unfixed mlx-swift), point
# the Package.swift deps back at ml-explore/mlx-swift and rebuild.
#
# Requirements: the Tesseract Agent app is RUNNING with the HTTP server enabled
# (scripts/dev.sh dev-release, then confirm `curl 127.0.0.1:8321/health`).
#
# Usage:
#   scripts/repro-image-cache-crash.sh [--iterations N] [--concurrent K]
#                                      [--delay SECONDS] [--alternate]
#                                      [--request FILE] [--port PORT]
#
#   --iterations N   How many rounds to send (default 300).
#   --concurrent K   Fire K identical requests in parallel each round to force
#                    overlapping restores of the same node (default 1).
#   --delay SECONDS  Sleep between rounds (default 0 — maximizes overlap with
#                    the post-request speculative prefill, the prime suspect).
#   --alternate      Rotate between TWO different image requests to force cache
#                    eviction/supersession (supersedeAncestorLeaves dropBody)
#                    between identical sends.
#   --request FILE   Replay this specific recorded request instead of auto-pick.
#   --port PORT      Server port (default 8321).
#
# Remove this script once ml-explore/mlx#3688 ships upstream and the fork pin is
# dropped (it depends on sandbox-recorded requests, so it is a manual dev tool,
# not a CI test).
set -uo pipefail

PORT=8321
ITERATIONS=300
CONCURRENT=1
DELAY=0
ALTERNATE=0
REQUEST_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations) ITERATIONS="$2"; shift 2;;
    --concurrent) CONCURRENT="$2"; shift 2;;
    --delay) DELAY="$2"; shift 2;;
    --alternate) ALTERNATE=1; shift;;
    --request) REQUEST_FILE="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

DIR="$HOME/Library/Containers/app.tesseract.agent/Data/tmp/tesseract-debug/http-completions"
BASE="http://127.0.0.1:${PORT}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

log() { echo "[DEBUG-imgcache] $*"; }

# --- preflight ---------------------------------------------------------------
if ! curl -sf -m 3 "${BASE}/health" >/dev/null; then
  echo "ERROR: server not reachable at ${BASE}/health — start the app first" >&2
  exit 1
fi
APP_PID="$(pgrep -f 'Tesseract Agent.app/Contents/MacOS/Tesseract Agent' | head -1)"
if [[ -z "${APP_PID:-}" ]]; then
  echo "ERROR: could not find the running app PID" >&2; exit 1
fi
log "server up, app PID=$APP_PID, port=$PORT"

# --- pick image-bearing recorded request(s) ---------------------------------
pick_image_requests() {
  # newest image-bearing recordings last; print full paths
  for f in $(ls -tr "$DIR"/*-request.json 2>/dev/null); do
    grep -q '"image_url"' "$f" && echo "$f"
  done
}

if [[ -n "$REQUEST_FILE" ]]; then
  PRIMARY="$REQUEST_FILE"; SECONDARY="$REQUEST_FILE"
else
  # bash 3.2 compatible (no mapfile / no negative indexing): capture once,
  # newest-last, then slice — avoids SIGPIPE on the producer loop.
  ALL_IMG="$(pick_image_requests)"
  PRIMARY="$(printf '%s\n' "$ALL_IMG" | tail -1)"   # newest image-bearing request
  SECONDARY="$(printf '%s\n' "$ALL_IMG" | head -1)" # oldest, a DIFFERENT image set
  if [[ -z "$PRIMARY" ]]; then
    echo "ERROR: no image-bearing recordings in $DIR — send an image in the app once" >&2
    exit 1
  fi
fi

# Strip the leading `// session=...` marker; keep the session for the header so
# cache routing matches the original (same-node hits + speculative scheduling).
prep_body() {  # $1 = recorded file -> writes body to $2, echoes session id
  local src="$1" out="$2"
  local sess
  sess="$(head -1 "$src" | sed -n 's#^// session=##p')"
  tail -n +2 "$src" > "$out"
  echo "$sess"
}

BODY_A="$TMP/a.json"; BODY_B="$TMP/b.json"
SESS_A="$(prep_body "$PRIMARY" "$BODY_A")"
SESS_B="$(prep_body "$SECONDARY" "$BODY_B")"
log "primary  = $(basename "$PRIMARY")  session=${SESS_A:-none}  $(du -h "$BODY_A" | cut -f1)"
[[ "$ALTERNATE" -eq 1 ]] && log "alternate= $(basename "$SECONDARY")  session=${SESS_B:-none}"

send() {  # $1 body, $2 session -> stream consumed and discarded
  curl -sN -m 120 -o /dev/null \
    -H 'Content-Type: application/json' \
    ${2:+-H "x-session-affinity: $2"} \
    --data-binary @"$1" \
    "${BASE}/v1/chat/completions"
}

alive() { kill -0 "$APP_PID" 2>/dev/null; }

dump_metal_error() {
  log "app DIED — pulling the Metal error from the unified log..."
  log show --last 3m --predicate "process == \"Tesseract Agent\"" --info --debug 2>/dev/null \
    | grep -iE "command buffer|MTLCommandBufferError|InvalidResource|Invalid Resource|check_error|IOGPU|out of memory|abort" \
    | tail -20
}

# --- loop --------------------------------------------------------------------
log "starting loop: iterations=$ITERATIONS concurrent=$CONCURRENT delay=$DELAY alternate=$ALTERNATE"
START_EPOCH=$SECONDS
for ((i=1; i<=ITERATIONS; i++)); do
  if ! alive; then log "crash detected BEFORE round $i"; dump_metal_error; exit 7; fi

  if [[ "$CONCURRENT" -gt 1 ]]; then
    pids=()
    for ((k=0; k<CONCURRENT; k++)); do send "$BODY_A" "$SESS_A" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
  else
    send "$BODY_A" "$SESS_A"
    if [[ "$ALTERNATE" -eq 1 ]]; then send "$BODY_B" "$SESS_B"; fi
  fi

  if ! alive; then
    log "CRASH after round $i (~$((SECONDS-START_EPOCH))s, ~$i sends)"
    dump_metal_error
    exit 7
  fi

  if (( i % 10 == 0 )); then log "round $i ok (~$((SECONDS-START_EPOCH))s elapsed)"; fi
  [[ "$DELAY" != "0" ]] && sleep "$DELAY"
done

log "completed $ITERATIONS rounds with NO crash — raise --iterations, try --concurrent 2/3, or --alternate"
exit 0
