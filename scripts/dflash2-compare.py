#!/usr/bin/env python3
"""Compare saved DFlash runs without paying for another AR generation."""

import argparse
import json
import statistics
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument(
        "--require-identity",
        action="store_true",
        help="require complete before/after token streams to match",
    )
    args = parser.parse_args()
    before, after = (json.loads(path.read_text()) for path in (args.before, args.after))
    for key in ("promptSHA256", "promptTokens", "maxNewTokens", "model"):
        if before[key] != after[key]:
            parser.error(f"incomparable {key}: {before[key]!r} != {after[key]!r}")
    if "inputTokenIds" in before and "inputTokenIds" in after:
        if before["inputTokenIds"] != after["inputTokenIds"]:
            parser.error("rendered prompt tokens differ")
    if args.require_identity and not all(
        report.get("capturedFullStream", report["fullIdentityCheck"])
        for report in (before, after)
    ):
        parser.error(
            "--require-identity needs reports containing complete token streams"
        )

    common = sorted(
        {r["arm"] for r in before["results"]} & {r["arm"] for r in after["results"]}
    )
    if not common:
        parser.error("no matching benchmark arms")
    failed = False
    for arm in common:
        left, right = (
            [r for r in report["results"] if r["arm"] == arm]
            for report in (before, after)
        )
        rates = [
            statistics.median(r["tokens"] / r["decodeSeconds"] for r in rows)
            for rows in (left, right)
        ]
        counts = [
            {(r["accepted"], r["proposed"], r["rounds"], r["tokens"]) for r in rows}
            for rows in (left, right)
        ]
        identity = "not checked"
        if args.require_identity:
            expected = left[0]["fingerprint"]
            if any(len(r["fingerprint"]) != r["tokens"] for r in left + right):
                parser.error(f"incomplete token stream for {arm}")
            match = all(r["fingerprint"] == expected for r in left + right)
            failed |= not match
            identity = "MATCH" if match else "DIVERGED"
        sys.stdout.write(
            f"{arm}: {rates[0]:.2f} -> {rates[1]:.2f} tok/s "
            f"({100 * (rates[1] / rates[0] - 1):+.1f}%), "
            f"round accounting {'MATCH' if counts[0] == counts[1] else 'CHANGED'}, "
            f"before/after identity {identity}\n"
        )
        if counts[0] != counts[1]:
            sys.stdout.write(
                "  Acceptance changed; throughput alone does not isolate round cost.\n"
            )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
