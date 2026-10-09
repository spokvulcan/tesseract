#!/usr/bin/env bash
set -euo pipefail

# The speed ruler: cold prefill on the frozen 2K/8K/32K prompts and 512
# greedy DFlash2 tokens on each decode fixture, one JSON report per run.
# Options pass through to the bench (benchmarks/dflash2/README.md):
#   --bench-prefill prefill-8k.txt   prefill prompts only these (none: skip)
#   --bench-fixtures summary,code    decode these fixtures only (none: skip)
#   --bench-check                    AR reference and full-stream identity
#   --bench-runs N                   DFlash2 runs per fixture
#   --bench-json PATH                the report
#   --no-build                       reuse the last Release build
# TESSERACT_BENCH_APP=<.app> runs another build (an A/B baseline).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/bench.sh" quick --model qwen3.8-27b --dflash2-bench --bench-ruler \
    --bench-fixture-dir "$SCRIPT_DIR/../benchmarks/dflash2" "$@"
