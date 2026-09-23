#!/bin/bash
# Finishes attempt 2 after it was stopped from outside during the PR's second
# probe (00:37 UTC): with the owner's go-ahead, that probe runs once more on
# a new SSD directory, then the PR-only checks, exactly as run_gate.sh would
# have run them.
set -u
REPO=/Users/owl/projects/tesseract
GATE=$REPO/benchmarks/allocation-profile/2026-09-22-gate
OUT=/Users/owl/bench-results/cache-claim-gate-2026-09-22-attempt2
PRODUCTS=/Users/owl/Library/Developer/Xcode/DerivedData
PR_APP="$PRODUCTS/tesseract-eqvahrkwytljlgelhybxketmldvq/Build/Products/Release/Tesseract Agent.app/Contents/MacOS/Tesseract Agent"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/gate.log"; }

settle() {
  sleep 30
  for _ in $(seq 1 60); do
    [ "$(sysctl -n kern.memorystatus_vm_pressure_level)" = "1" ] && return 0
    sleep 5
  done
  return 1
}

run() {
  local name=$1
  shift
  if ! settle; then
    log "$name not started: memory pressure stayed above normal"
    return
  fi
  log "start $name"
  "$@" > "$OUT/$name.wrapper.log" 2>&1
  log "end $name exit=$?"
}

cd "$REPO"
log "finishing attempt 2; PR $(shasum -a 256 "$PR_APP" | cut -c1-12)"
run run4-pr-probe python3 scripts/allocation_inventory_probe.py \
  --app "$PR_APP" --output "$OUT/run4-pr-probe" --run --ssd-directory "$OUT/run4-pr-probe-ssd"
run pr-parity python3 scripts/bounded_cache_parity.py --app "$PR_APP" --output "$OUT/pr-parity" --run
export TESSERACT_E2E_SPECULATION=dflash2
mkdir -p "$OUT/pr-e2e-ssd"
run pr-e2e python3 "$GATE/watchdog.py" --output "$OUT/pr-e2e" -- \
  "$PR_APP" --prefix-cache-e2e --bench-model-id qwen3.8-27b --bench-output "$OUT/pr-e2e-harness" \
  -prefixCacheSSDDirectoryOverride "$OUT/pr-e2e-ssd"
log "gate ends"
