#!/usr/bin/env python3
"""Rotated Ternary Checkpoint parity reference (ADR-0067).

The reference half of `scripts/dev.sh rotated-checkpoint-parity`. The Swift
half (`RotatedCheckpointParityRunner`, `--rotated-checkpoint-parity`) loads the
pack through the production path, greedy-decodes a fixed prompt and writes the
prompt and generated token ids to `latest.json`. This script decodes the same
prompt ids through mlx-vlm's own `prism_hadamard_qwen35` implementation and
scores the two:

1. greedy agreement — longest common prefix of the two greedy continuations
   (float differences between two engines eventually fork a greedy trajectory,
   so this alone is a weak signal);
2. teacher-forced agreement — the Swift continuation is fed back through the
   reference in one pass and the reference argmax at every position is
   compared with the token Swift actually emitted. A missing rotation scores
   near zero here; engine-level float noise costs a token or two.

PASS needs a common prefix of at least --min-prefix tokens and a teacher-forced
agreement of at least --min-agreement.

mlx-vlm loads the pack as stored: float32 norms, taps and gated-delta projections,
which MLX promotes the residual stream to after the first layer. The app casts
those tensors to the manifest's float16 at load (the cast mlx-vlm's converter
applies to a checkpoint it writes), so --match-app-dtypes applies the same cast
here and isolates the rotation logic from that half-precision difference. Run
both: the untouched reference shows what the cast costs, the matched one whether
the rotation is right.

Usage (from the repo root, venv at research/bonsai-venv with mlx-vlm installed):

    research/bonsai-venv/bin/python scripts/rotated_checkpoint_reference.py \
        --report "$(getconf DARWIN_USER_TEMP_DIR)tesseract-debug/benchmark/rotated-checkpoint-parity/latest.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx


def greedy(model, prompt_ids: list[int], count: int) -> list[int]:
    from mlx_vlm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model.language_model)
    y = mx.array(prompt_ids)[None]
    out: list[int] = []
    for _ in range(count):
        logits = model.language_model(y, cache=cache).logits[:, -1, :]
        token = int(mx.argmax(logits, axis=-1).item())
        out.append(token)
        y = mx.array([[token]])
    return out


def teacher_forced(model, prompt_ids: list[int], generated: list[int]) -> list[int]:
    ids = mx.array(prompt_ids + generated)[None]
    logits = model.language_model(ids).logits[0]
    start = len(prompt_ids) - 1
    return mx.argmax(logits[start : start + len(generated)], axis=-1).tolist()


def cast_unpacked_to_float16(model, config: dict) -> int:
    """Mirror `HadamardQuantizedManifest.castingUnpackedWeights`: every floating
    parameter except the packed modules' own and `A_log` goes to float16."""
    from mlx.utils import tree_flatten, tree_unflatten

    packed = {"language_model." + m["path"] for m in config["modules"]}
    updates = []
    for key, value in tree_flatten(model.parameters()):
        module_path = key.rsplit(".", 1)[0]
        if (
            mx.issubdtype(value.dtype, mx.floating)
            and value.dtype != mx.float16
            and not key.endswith("A_log")
            and module_path not in packed
        ):
            updates.append((key, value.astype(mx.float16)))
    if updates:
        model.update(tree_unflatten(updates))
        mx.eval(model.parameters())
    return len(updates)


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report", required=True, type=Path, help="latest.json from the Swift half")
    parser.add_argument("--model-dir", type=Path, help="override the model directory recorded in the report")
    parser.add_argument("--min-prefix", type=int, default=16)
    parser.add_argument("--min-agreement", type=float, default=0.9)
    parser.add_argument("--match-app-dtypes", action="store_true",
                        help="cast the unpacked float32 tensors to float16 as the app does")
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    prompt_ids = [int(t) for t in report["promptTokens"]]
    swift_ids = [int(t) for t in report["generatedTokens"]]
    model_dir = args.model_dir or Path(report["modelDir"])
    if not swift_ids:
        print("FAIL: the report carries no generated tokens", file=sys.stderr)
        return 1

    from mlx_vlm.utils import load

    print(f"model: {report['model']} ({model_dir})")
    print(f"swift rotated modules: {report['rotatedLinearModules']} linear, "
          f"{report['rotatedEmbeddingModules']} embedding")
    model, processor = load(str(model_dir))
    tokenizer = getattr(processor, "tokenizer", processor)
    if args.match_app_dtypes:
        config = json.loads((model_dir / "config.json").read_text())
        print(f"cast {cast_unpacked_to_float16(model, config)} unpacked tensors to float16")

    reference_ids = greedy(model, prompt_ids, len(swift_ids))
    prefix = common_prefix(swift_ids, reference_ids)
    forced = teacher_forced(model, prompt_ids, swift_ids)
    agreed = sum(1 for a, b in zip(forced, swift_ids) if a == b)
    agreement = agreed / len(swift_ids)

    print(f"swift:     {tokenizer.decode(swift_ids)!r}")
    print(f"reference: {tokenizer.decode(reference_ids)!r}")
    print(f"greedy common prefix: {prefix}/{len(swift_ids)} tokens")
    print(f"teacher-forced agreement: {agreed}/{len(swift_ids)} = {agreement:.3f}")
    if prefix < len(swift_ids):
        print(f"first greedy divergence at token {prefix}: "
              f"swift={swift_ids[prefix]} reference={reference_ids[prefix]}")

    ok = prefix >= args.min_prefix and agreement >= args.min_agreement
    print(f"Overall: {'PASS' if ok else 'FAIL'} "
          f"(need prefix >= {args.min_prefix}, agreement >= {args.min_agreement})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
