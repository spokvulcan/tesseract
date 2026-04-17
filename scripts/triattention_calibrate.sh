#!/usr/bin/env bash
# Regenerate a TriAttention calibration artifact for a PARO checkpoint.
#
# Two supported flows:
#
#   (a) Re-key an upstream stats .pt (produced by
#       https://github.com/WeianMao/triattention scripts/calibrate.py on a
#       GPU box or an fp16 checkpoint):
#
#         scripts/triattention_calibrate.sh rekey \
#           --model-dir ~/Library/Containers/app.tesseract.agent/Data/Library/Application\ Support/models/z-lab_Qwen3.5-4B-PARO \
#           --stats-pt /tmp/upstream-calibration.pt
#
#   (b) Run HF-transformers calibration locally (Apple Silicon, MPS).
#       Requires `torch` + `transformers` — not compatible with paroquant
#       weights as of writing. Point at an fp16 copy of the checkpoint.
#
#         scripts/triattention_calibrate.sh calibrate \
#           --model-dir /path/to/fp16-copy \
#           --input scripts/triattention_calibration_text.txt
#
# The output lands in tesseract/Resources/TriAttention/v1/<fingerprint>.pt
# and is picked up by the Xcode synchronized group at the next build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_OUTPUT="$REPO_ROOT/TriAttention/v1"
DEFAULT_CALIB_TEXT="$SCRIPT_DIR/triattention_calibration_text.txt"
VENV_PYTHON="$SCRIPT_DIR/.triattention-venv/bin/python3"
# Prefer the venv created by this script's first run; fall back to the
# ambient python if `scripts/.triattention-venv` hasn't been set up yet.
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="${PYTHON:-$VENV_PYTHON}"
else
    PYTHON="${PYTHON:-python3}"
fi

usage() {
    sed -n '3,28p' "$0"
    exit 2
}

MODE="${1:-}"
shift || usage

case "$MODE" in
    rekey|calibrate)
        ;;
    -h|--help|"")
        usage
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage
        ;;
esac

MODEL_DIR=""
STATS_PT=""
INPUT=""
OUTPUT="$DEFAULT_OUTPUT"

while [ $# -gt 0 ]; do
    case "$1" in
        --model-dir)   MODEL_DIR="$2"; shift 2 ;;
        --stats-pt)    STATS_PT="$2"; shift 2 ;;
        --input)       INPUT="$2"; shift 2 ;;
        --output)      OUTPUT="$2"; shift 2 ;;
        -h|--help)     usage ;;
        *)             echo "Unknown arg: $1" >&2; usage ;;
    esac
done

if [ -z "$MODEL_DIR" ]; then
    echo "--model-dir is required" >&2
    exit 2
fi

ARGS=(--model-dir "$MODEL_DIR" --output "$OUTPUT")

case "$MODE" in
    rekey)
        if [ -z "$STATS_PT" ]; then
            echo "rekey mode requires --stats-pt" >&2
            exit 2
        fi
        ARGS+=(--stats-pt "$STATS_PT")
        ;;
    calibrate)
        if [ -z "$INPUT" ]; then
            INPUT="$DEFAULT_CALIB_TEXT"
        fi
        ARGS+=(--input "$INPUT")
        ;;
esac

exec "$PYTHON" "$SCRIPT_DIR/triattention_calibrate.py" "${ARGS[@]}"
