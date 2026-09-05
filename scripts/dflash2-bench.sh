#!/usr/bin/env bash
set -euo pipefail

# Fast iteration: one DFlash pass, no AR pass. Add --bench-check periodically
# for a full-stream comparison, --bench-runs N for repeatability, and
# --no-build when changing only prompts/widths/lengths. Other args pass through.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ " ${*:-} " == *" --help "* ]]; then
    cat <<'USAGE'
Usage: scripts/dflash2-bench.sh [options]
  --no-build                    Reuse the last Release binary
  --bench-prompt-file PATH       Frozen user prompt (default: short travel fixture)
  --bench-max-tokens N           Generated token limit (default: 192)
  --bench-blocks 3,5,8           Sequential width sweep (default: 8)
  --bench-runs N                 Repeats per width (default: 1)
  --bench-check                  Add AR and check the full token stream
  --bench-round-timings          Record individual rounds in the JSON report
  --bench-draft-policy POLICY    Experimental precision (default: 4bit)
  --bench-json PATH              Save a machine-readable report
USAGE
    exit 0
fi
# A short prose fixture keeps the default loop out of the 32-second summary
# prefill. Explicit fixtures and legacy workload controls take precedence.
USE_DEFAULT_PROMPT=1
for arg in "$@"; do
    case "$arg" in
        --bench-prompt-file|--bench-prompt-variants|--bench-context-mult) USE_DEFAULT_PROMPT=0 ;;
    esac
done
if [[ -n "${DFLASH2_BENCH_PROMPT_FILE:-}${DFLASH2_BENCH_PROMPT:-}" ]]; then
    USE_DEFAULT_PROMPT=0
fi
if [ "$USE_DEFAULT_PROMPT" = 1 ]; then
    set -- --bench-prompt-file "$SCRIPT_DIR/../benchmarks/dflash2/travel.txt" "$@"
fi
exec "$SCRIPT_DIR/bench.sh" quick --model qwen3.8-27b --dflash2-bench --bench-fast "$@"
