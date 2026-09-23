#!/bin/bash
# The #554 gate in its committed order: main, PR, main, PR, then the PR-only
# checks. Every campaign runs once, against its own new SSD cache directory,
# so each starts cold and the owner's cache is never read or written. A stop
# is recorded in that campaign's own outcome; the next campaign still starts
# only from normal memory pressure.
set -u
REPO=/Users/owl/projects/tesseract
GATE=$REPO/benchmarks/allocation-profile/2026-09-22-gate
OUT=/Users/owl/bench-results/cache-claim-gate-2026-09-22-attempt2
PRODUCTS=/Users/owl/Library/Developer/Xcode/DerivedData
MAIN_APP="$PRODUCTS/tesseract-hdggevinvbflgsftwpanblrtnizc/Build/Products/Release/Tesseract Agent.app/Contents/MacOS/Tesseract Agent"
PR_APP="$PRODUCTS/tesseract-eqvahrkwytljlgelhybxketmldvq/Build/Products/Release/Tesseract Agent.app/Contents/MacOS/Tesseract Agent"

mkdir -p "$OUT"
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
log "gate begins; main $(shasum -a 256 "$MAIN_APP" | cut -c1-12), PR $(shasum -a 256 "$PR_APP" | cut -c1-12)"
for spec in 1:main 2:pr 3:main 4:pr; do
  n=${spec%%:*}
  arm=${spec##*:}
  app=$MAIN_APP
  [ "$arm" = pr ] && app=$PR_APP
  run "run$n-$arm-profile" python3 scripts/cancelled_generation_profile.py \
    --app "$app" --plan "$GATE/plan-$arm.json" --output "$OUT/run$n-$arm-profile" \
    --ssd-directory "$OUT/run$n-$arm-profile-ssd"
  if [ "$n" = 1 ] && [ -z "$(ls -A "$OUT/run1-main-profile-ssd" 2>/dev/null)" ]; then
    log "the first campaign wrote nothing to its SSD directory; stopping the gate"
    exit 1
  fi
  run "run$n-$arm-probe" python3 scripts/allocation_inventory_probe.py \
    --app "$app" --output "$OUT/run$n-$arm-probe" --run --ssd-directory "$OUT/run$n-$arm-probe-ssd"
done
run pr-parity python3 scripts/bounded_cache_parity.py --app "$PR_APP" --output "$OUT/pr-parity" --run
export TESSERACT_E2E_SPECULATION=dflash2
mkdir -p "$OUT/pr-e2e-ssd"
run pr-e2e python3 "$GATE/watchdog.py" --output "$OUT/pr-e2e" -- \
  "$PR_APP" --prefix-cache-e2e --bench-model-id qwen3.8-27b --bench-output "$OUT/pr-e2e-harness" \
  -prefixCacheSSDDirectoryOverride "$OUT/pr-e2e-ssd"
log "gate ends"
