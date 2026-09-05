#!/usr/bin/env python3
"""Isolate recurrent replay work; requires Python MLX and the recorded vendor commit.

Run with no other GPU workload. This measures 48 dependent replays, not
whole-model generation. The candidate preserves the original state arithmetic.
"""
from pathlib import Path
import json
import statistics
import subprocess
import sys
import time
import mlx.core as mx

vendor = Path(__file__).resolve().parents[2] / "Vendor" / "mlx-swift-lm"
baseline = "6d251c6cedff52ebdf9872129b32bebb5b2f9c32"
original = subprocess.check_output(
    [
        "git",
        "-C",
        str(vendor),
        "show",
        baseline + ":Libraries/MLXLMCommon/GatedDelta.swift",
    ],
    text=True,
)
src = original.split('let source = """', 1)[1].split('"""', 1)[0]
src = src.replace("\\(maskSource)", "mask[b_idx * T + t]")
state_src = src
for line in [
    "auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;",
    "y += b_idx * T * Hv * Dv + hv_idx * Dv;",
    "float out = 0.0f;",
    "out += state[i] * q_[s_idx];",
    "out = simd_sum(out);",
    "q_ += Hk * Dk;",
    "y += Hv * Dv;",
]:
    state_src = state_src.replace(line, "")
state_src = state_src.replace(
    "if (thread_index_in_simdgroup == 0) {\n                  y[dv_idx] = static_cast<InT>(out);\n                }",
    "",
)
state_src = state_src.replace(
    "else {\n                y[dv_idx] = static_cast<InT>(0);\n              }", ""
)
base = mx.fast.metal_kernel(
    name="replay_probe_base",
    input_names=["q", "k", "v", "g", "beta", "state_in", "T", "mask"],
    output_names=["y", "state_out"],
    source=src,
)
state = mx.fast.metal_kernel(
    name="replay_probe_state",
    input_names=["k", "v", "g", "beta", "state_in", "T", "mask"],
    output_names=["state_out"],
    source=state_src,
)
mx.random.seed(42)
B, T, Hk, Dk, Hv, Dv = 1, 8, 16, 128, 48, 128
q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16) * 0.03
k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16) * 0.03
v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
g = mx.sigmoid(mx.random.normal((B, T, Hv)))
beta = mx.sigmoid(mx.random.normal((B, T, Hv)))
s = mx.random.normal((B, Hv, Dv, Dk))
t = mx.array(T)
mx.eval(q, k, v, g, beta, s, t)
kw = dict(
    template=[
        ("InT", mx.bfloat16),
        ("StT", mx.float32),
        ("Dk", Dk),
        ("Dv", Dv),
        ("Hk", Hk),
        ("Hv", Hv),
    ],
    grid=(32, Dv, B * Hv),
    threadgroup=(32, 4, 1),
)


def run(which, count, initial):
    mask = mx.arange(T)[None, :] < count
    if which == "base":
        return base(
            inputs=[q, k, v, g, beta, initial, t, mask],
            output_shapes=[(B, T, Hv, Dv), initial.shape],
            output_dtypes=[mx.bfloat16, mx.float32],
            **kw,
        )[1]
    return state(
        inputs=[k, v, g, beta, initial, t, mask],
        output_shapes=[initial.shape],
        output_dtypes=[mx.float32],
        **kw,
    )[0]


for count in range(9):
    a, b = run("base", count, s), run("state", count, s)
    assert mx.array_equal(a, b).item(), (count, mx.max(mx.abs(a - b)).item())
sys.stdout.write("All accepted-prefix counts 0..8: bitwise exact\n")
for count in [1, 3, 8]:
    times = {"base": [], "state": []}
    for rep in range(12):
        for which in ["base", "state"] if rep % 2 == 0 else ["state", "base"]:
            x = s
            tic = time.perf_counter()
            for _ in range(48):
                x = run(which, count, x)
            mx.eval(x)
            times[which].append((time.perf_counter() - tic) * 1000)
    sys.stdout.write(
        json.dumps(
            {
                "valid": count,
                "ms_per_48_replays": {
                    name: statistics.median(vals) for name, vals in times.items()
                },
            }
        )
        + "\n"
    )
