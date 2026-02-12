#!/usr/bin/env bash
# Headless Qwen3-TTS runner for console workflows.
# Builds and runs Vendor/mlx-audio-swift's mlx-audio-swift-tts executable.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PKG_DIR="$ROOT_DIR/Vendor/mlx-audio-swift"
PRODUCT_NAME="mlx-audio-swift-tts"
DEFAULT_MODEL="mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
DERIVED_DATA_DIR="${DERIVED_DATA_DIR:-$HOME/Library/Developer/Xcode/DerivedData}"

bin_dir() {
    swift build --package-path "$PKG_DIR" --product "$PRODUCT_NAME" --show-bin-path
}

bin_path() {
    printf "%s/%s" "$(bin_dir)" "$PRODUCT_NAME"
}

abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf "%s" "$path"
    else
        printf "%s/%s" "$PWD" "$path"
    fi
}

prepare_output_path() {
    local path
    path="$(abs_path "$1")"
    mkdir -p "$(dirname "$path")"
    printf "%s" "$path"
}

build_cli() {
    swift build --package-path "$PKG_DIR" --product "$PRODUCT_NAME"
}

ensure_built() {
    local bin
    bin="$(bin_path)"
    if [[ ! -x "$bin" ]]; then
        build_cli
    fi
}

latest_deriveddata_metallib() {
    local latest_path=""
    local latest_mtime=0
    local path
    while IFS= read -r -d '' path; do
        local mtime
        mtime="$(stat -f "%m" "$path" 2>/dev/null || echo 0)"
        if (( mtime > latest_mtime )); then
            latest_mtime="$mtime"
            latest_path="$path"
        fi
    done < <(find "$DERIVED_DATA_DIR" -path "*/Build/Products/Debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" -print0 2>/dev/null)
    printf "%s" "$latest_path"
}

resolve_metallib_source() {
    local local_copy
    local_copy="$(bin_dir)/default.metallib"
    if [[ -f "$local_copy" ]]; then
        printf "%s" "$local_copy"
        return 0
    fi

    if [[ -n "${MLX_DEFAULT_METALLIB:-}" ]]; then
        if [[ -f "$MLX_DEFAULT_METALLIB" ]]; then
            printf "%s" "$MLX_DEFAULT_METALLIB"
            return 0
        fi
        echo "MLX_DEFAULT_METALLIB is set but file does not exist: $MLX_DEFAULT_METALLIB" >&2
        exit 1
    fi

    local found
    found="$(latest_deriveddata_metallib)"
    if [[ -n "$found" ]]; then
        printf "%s" "$found"
        return 0
    fi

    cat >&2 <<EOF
Could not find default.metallib for MLX.
Run one of these first:
  - xcodebuild build -project tesseract.xcodeproj -scheme tesseract
  - export MLX_DEFAULT_METALLIB=/absolute/path/to/default.metallib
EOF
    exit 1
}

stage_metallib() {
    local source_path
    source_path="$(resolve_metallib_source)"
    local destination_path
    destination_path="$(bin_dir)/default.metallib"

    if [[ "$source_path" != "$destination_path" ]]; then
        cp -f "$source_path" "$destination_path"
    fi
}

has_model_arg() {
    local arg
    for arg in "$@"; do
        case "$arg" in
            --model|--model=*)
                return 0
                ;;
        esac
    done
    return 1
}

normalize_args() {
    local -a input=("$@")
    local -a output=()
    local i=0
    while (( i < ${#input[@]} )); do
        local arg="${input[$i]}"
        case "$arg" in
            --output|-o|--ref_audio)
                output+=("$arg")
                ((i++))
                if (( i >= ${#input[@]} )); then
                    echo "Missing value for $arg" >&2
                    exit 1
                fi
                if [[ "$arg" == "--output" || "$arg" == "-o" ]]; then
                    output+=("$(prepare_output_path "${input[$i]}")")
                else
                    output+=("$(abs_path "${input[$i]}")")
                fi
                ;;
            --output=*|--ref_audio=*)
                local key="${arg%%=*}"
                local value="${arg#*=}"
                if [[ "$key" == "--output" ]]; then
                    output+=("$key=$(prepare_output_path "$value")")
                else
                    output+=("$key=$(abs_path "$value")")
                fi
                ;;
            *)
                output+=("$arg")
                ;;
        esac
        ((i++))
    done
    printf "%s\n" "${output[@]}"
}

cmd_build() {
    build_cli
    stage_metallib
    echo "Built $PRODUCT_NAME"
    echo "Binary: $(bin_path)"
    echo "Metallib: $(bin_dir)/default.metallib"
}

cmd_run() {
    ensure_built
    stage_metallib

    local -a args=("$@")
    if ! has_model_arg "${args[@]}"; then
        args=(--model "$DEFAULT_MODEL" "${args[@]}")
    fi

    local -a normalized=()
    while IFS= read -r line; do
        normalized+=("$line")
    done < <(normalize_args "${args[@]}")

    local dir
    dir="$(bin_dir)"
    local bin
    bin="$(bin_path)"

    (
        cd "$dir"
        "$bin" "${normalized[@]}"
    )
}

cmd_smoke() {
    local out_path="${1:-$ROOT_DIR/.artifacts/headless-smoke.wav}"
    mkdir -p "$(dirname "$out_path")"
    cmd_run \
        --text "Headless smoke test. Audio quality check." \
        --output "$out_path" \
        --max_tokens 96 \
        --temperature 0.7 \
        --top_p 0.9

    if [[ ! -s "$out_path" ]]; then
        echo "Smoke test failed: no WAV output at $out_path" >&2
        exit 1
    fi
    echo "Smoke test passed: $out_path"
}

usage() {
    cat <<EOF
Usage: scripts/tts-headless.sh <command> [args...]

Commands:
  build                  Build headless TTS CLI and stage default.metallib
  run [tts args...]      Run headless TTS CLI (defaults model to Qwen3 if omitted)
  smoke [output.wav]     Generate a short WAV and verify output file exists
  help                   Show this help

Examples:
  scripts/tts-headless.sh build
  scripts/tts-headless.sh run --text "Hello from terminal" --output /tmp/tts.wav
  scripts/tts-headless.sh smoke
EOF
}

case "${1:-}" in
    build)
        shift
        cmd_build "$@"
        ;;
    run)
        shift
        cmd_run "$@"
        ;;
    smoke|test)
        shift
        cmd_smoke "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
esac
