#!/usr/bin/env python3
"""Apply the 2026-09-19 pre-registration's per-case rules to the runner's
immediately-saved observations (observations.jsonl, dequantize.jsonl,
fidelity.jsonl) and write verdicts.json. The runner's own report writer
failed on a JSON encoding defect (NaN for the control arm's undefined paired
excess) after every observation had been saved; this script computes the
same verdicts from the same records. It selects nothing: the rules are the
pre-registered ones and every sample is used."""

import json
import pathlib
import statistics
import sys

report = pathlib.Path(sys.argv[1])
plan = json.loads((report / "approved-plan.json").read_text())


def rows(name):
    return [json.loads(line) for line in (report / name).read_text().splitlines() if line.strip()]


observations = rows("observations.jsonl")
dequantize = rows("dequantize.jsonl")
copies = rows("copy.jsonl")
fidelity = rows("fidelity.jsonl")
blocks = len(plan["orders"])


def p95(values):
    values = sorted(values)
    rank = -(-((len(values) - 1) * 95) // 100)
    return values[min(max(rank, 0), len(values) - 1)]


verdicts = []
for case in plan["cases"]:
    case_id = case["id"]
    case_obs = [o for o in observations if o["caseID"] == case_id]
    control = {o["block"]: o for o in case_obs if o["arm"] == "fp16"}
    control_fid = {f["block"]: f for f in fidelity if f["caseID"] == case_id and f["arm"] == "fp16"}
    for arm in ("fp16", "8", "4"):
        arm_obs = [o for o in case_obs if o["arm"] == arm]
        arm_fid = [f for f in fidelity if f["caseID"] == case_id and f["arm"] == arm]
        paired = [(o, control[o["block"]]) for o in arm_obs if o["block"] in control]
        ttft = [o["ttftSeconds"] * 1000 for o in arm_obs]
        excess = [] if arm == "fp16" else [(w["ttftSeconds"] - c["ttftSeconds"]) * 1000 for w, c in paired]
        token_mismatch_blocks = [] if arm == "fp16" else [
            w["block"] for w, c in paired
            if w["generatedText"] != c["generatedText"] or w["generatedTokenIDs"] != c["generatedTokenIDs"]
            or w["generationTokenCount"] != c["generationTokenCount"]]
        prompt_mismatches = 0 if arm == "fp16" else sum(1 for w, c in paired if w["promptTokenCount"] != c["promptTokenCount"])
        deq = [s * 1000 for m in dequantize if m["caseID"] == case_id and m["arm"] == arm for s in m["seconds"]]
        overhead = [s * 1000 for m in dequantize if m["caseID"] == case_id and m["arm"] == arm for s in m["observerOverheadSeconds"]]
        mismatches = sum(f["mismatches"] for f in arm_fid)
        boundaries = sum(f["boundaries"] for f in arm_fid)
        coverage = all(f["boundaryKinds"] == control_fid[f["block"]]["boundaryKinds"] and f["boundaries"] > 0
                       for f in arm_fid if f["block"] in control_fid)
        invalid = [(o["block"], o["invalid"]) for o in arm_obs if o["invalid"]]
        control_invalid = [(o["block"], o["invalid"]) for o in control.values() if o["invalid"]]
        excess_median = statistics.median(excess) if excess else None
        deq_median = statistics.median(deq) if deq else None
        timing = None if arm == "fp16" or deq_median is None else excess_median <= deq_median
        if arm == "fp16":
            verdict = "REFERENCE" if not invalid and mismatches == 0 and boundaries > 0 else "INVALID"
        elif (len(arm_obs) != blocks or len(paired) != blocks or len(deq) != blocks * 6 or invalid or control_invalid):
            verdict = "INCONCLUSIVE"
        elif mismatches == 0 and boundaries > 0 and coverage and not token_mismatch_blocks and prompt_mismatches == 0 and timing:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        verdicts.append({
            "caseID": case_id, "arm": arm, "samples": len(arm_obs),
            "fidelityMismatches": mismatches, "fidelityBoundaries": boundaries,
            "boundaryKindCoverageMatchesControl": coverage,
            "tokenMismatchBlocks": token_mismatch_blocks, "promptTokenMismatches": prompt_mismatches,
            "ttftMs": [round(t, 1) for t in ttft], "ttftMedianMs": round(statistics.median(ttft), 1),
            "ttftP95Ms": round(p95(ttft), 1),
            "controlTtftMedianMs": round(statistics.median([o["ttftSeconds"] * 1000 for o in control.values()]), 1),
            "pairedExcessMs": [round(e, 1) for e in excess],
            "pairedExcessMedianMs": None if excess_median is None else round(excess_median, 1),
            "dequantizeMs": [round(d, 2) for d in deq],
            "dequantizeMedianMs": None if deq_median is None else round(deq_median, 2),
            "observerOverheadMedianMs": None if not overhead else round(statistics.median(overhead), 3),
            "copyMedianMs": None if arm != "fp16" else round(statistics.median(
                [s * 1000 for m in copies if m["caseID"] == case_id for s in m["seconds"]]), 2),
            "timingPasses": timing, "invalidObservations": invalid,
            "residentForm": arm_obs[0]["residentFormBeforeHit"], "residentBits": arm_obs[0].get("residentBitsBeforeHit"),
            "residentBodyBytes": arm_obs[0]["residentBodyBytesBeforeHit"],
            "restoreOffset": arm_obs[0]["cachedTokenCount"], "promptTokens": arm_obs[0]["promptTokenCount"],
            "serverRestoreMedianMs": round(statistics.median([o["serverRestoreMs"] for o in arm_obs]), 1),
            "mlxPeakDuringHitMax": max(o["mlxPeakDuringHit"] for o in arm_obs),
            "footprintAfterHitMax": max(o["footprintAfterHit"] or 0 for o in arm_obs),
            "verdict": verdict})

(report / "verdicts.json").write_text(json.dumps(verdicts, indent=2))
for v in verdicts:
    print(v["caseID"], v["arm"], v["verdict"], "tokens-mismatch", v["tokenMismatchBlocks"], "excess", v["pairedExcessMedianMs"], "deq", v["dequantizeMedianMs"], "fid", f'{v["fidelityMismatches"]}/{v["fidelityBoundaries"]}')
