#!/usr/bin/env python3
"""Run the #532 SSD read-path experiment under the owner plan's resource bounds.

The plan (`plan.json`, committed before the run) carries both the harness's
own fields (`cacheRoot`, `snapshotID`, `segmentSHA256`, `maxSegmentBytes`,
`maxMLXBytes`, `ownerApproval`) and the external bounds this wrapper
enforces: it refuses anything not APPROVED, verifies the release binary and
the exported chain files against their recorded SHA-256, samples the harness
process every `sampleIntervalMilliseconds`, terminates it on the first
breach and never retries. The harness itself bounds MLX memory; process
footprint, available memory, swap, pressure, disk and wall time are bounded
here, as the pre-registration requires.
"""

import argparse
import ctypes
import datetime
import json
import os
import pathlib
import subprocess
import time

from allocation_inventory_probe import OSProbe, file_digest, write_json
from warm_body_parity import available_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("status") != "APPROVED":
        parser.error("Plan status is not APPROVED")
    for key in ("ownerApproval", "appCommit", "vendorCommit", "releaseBinarySHA256", "cacheRoot",
                "snapshotID", "segmentSHA256", "privateOutputDirectory"):
        if not plan.get(key):
            parser.error(f"Plan field {key} is empty: not executable")
    resources = plan["resources"]
    for key, value in resources.items():
        if value is None:
            parser.error(f"Plan resource {key} is null: not executable")
    if args.output.exists():
        parser.error("Output must be a new directory")
    if subprocess.run(["pgrep", "-x", "Tesseract Agent"], capture_output=True).returncode == 0:
        parser.error("Quit the ordinary app before running the experiment")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(args.plan)], capture_output=True).returncode:
        parser.error("Commit the plan before running")
    app = pathlib.Path(plan["releaseBinary"])
    binary_digest = file_digest(app)
    if binary_digest != plan["releaseBinarySHA256"]:
        parser.error("Release binary SHA-256 differs from the plan")
    cache_root = pathlib.Path(plan["cacheRoot"])
    for relative, digest in plan["segmentSHA256"].items():
        if file_digest(cache_root / relative) != digest:
            parser.error(f"Chain file {relative} differs from the plan")
    model_dir = pathlib.Path(plan["model"]["directory"])
    if file_digest(model_dir / "config.json") != plan["model"]["configSHA256"]:
        parser.error("Model config.json differs from the plan")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    probe = OSProbe()
    page_size = probe.sysctl("vm.pagesize", ctypes.c_uint32()).value
    before = probe.read(os.getpid())
    initial_available = available_bytes(probe, page_size)
    if initial_available < resources["minimumInitialAvailableBytes"]:
        parser.error(f"Initial available memory {initial_available} below the plan minimum")
    if before["pressureLevel"] not in (1, 2):
        parser.error("Starting memory pressure must be normal or warning")
    args.output.mkdir(parents=True)
    args.output = args.output.resolve()
    stat = os.statvfs(str(args.output))
    disk_free = stat.f_bavail * stat.f_frsize
    if disk_free < resources["minimumInitialDiskFreeBytes"]:
        parser.error("Initial free disk below the plan minimum")

    command = [str(app), "--ssd-read-bench", "--ssd-read-plan", str(args.plan.resolve()),
               "--bench-model", str(model_dir), "--bench-model-id", plan["model"]["id"],
               "--bench-source-revision", head, "--bench-output", str(args.output / "harness"),
               "-isServerEnabled", "NO"]
    write_json(args.output / "environment.json", {
        "plan": plan, "planSHA256": file_digest(args.plan), "command": command,
        "binarySHA256": binary_digest, "sourceRevision": head,
        "workingTreeDirty": subprocess.run(["git", "diff", "--quiet", "HEAD"], capture_output=True).returncode != 0,
        "os": subprocess.check_output(["sw_vers"], text=True),
        "physicalMemoryBytes": probe.sysctl("hw.memsize", ctypes.c_uint64()).value,
        "initialAvailableBytes": initial_available, "initialDiskFreeBytes": disk_free,
        "baselineSystemSwapUsedBytes": before["systemSwapUsedBytes"],
        "startUTC": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    for name in ("ssd_read_bench.py", "allocation_inventory_probe.py", "warm_body_parity.py"):
        (args.output / name).write_bytes(pathlib.Path(__file__).with_name(name).read_bytes())

    samples = []
    stop = None
    started = time.monotonic()
    interval = resources["sampleIntervalMilliseconds"] / 1000
    with (args.output / "app.log").open("wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                try:
                    sample = probe.read(child.pid)
                except OSError:
                    if child.poll() is not None:
                        break
                    raise
                sample["availableBytes"] = available_bytes(probe, page_size)
                sample["monotonicSeconds"] = time.monotonic()
                samples.append(sample)
                swap_growth = sample["systemSwapUsedBytes"] - before["systemSwapUsedBytes"]
                stat = os.statvfs(str(args.output))
                if sample["pressureLevel"] >= resources["pressureStopLevel"] or sample["pressureLevel"] < 1:
                    stop = f"memory pressure level {sample['pressureLevel']}"
                elif sample["processFootprintBytes"] >= resources["footprintStopBytes"]:
                    stop = f"process footprint {sample['processFootprintBytes']}"
                elif sample["availableBytes"] < resources["minimumAvailableStopBytes"]:
                    stop = f"available memory {sample['availableBytes']}"
                elif swap_growth >= resources["swapGrowthStopBytes"]:
                    stop = f"system swap growth {swap_growth}"
                elif stat.f_bavail * stat.f_frsize < resources["minimumDiskFreeStopBytes"]:
                    stop = "free disk"
                elif time.monotonic() - started >= resources["maximumCampaignSeconds"]:
                    stop = "campaign deadline"
                if stop:
                    break
                time.sleep(interval)
        except Exception as error:
            stop = f"sampler error: {error}"
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=resources["cancelGraceSeconds"])
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    write_json(args.output / "os-samples.json", samples)
    report_dirs = list((args.output / "harness").glob("ssd-read-*"))
    records = None
    if len(report_dirs) == 1 and (report_dirs[0] / "records.json").exists():
        records = json.loads((report_dirs[0] / "records.json").read_text())
    outcome = {"exitCode": child.returncode, "resourceStop": stop,
               "elapsedSeconds": time.monotonic() - started, "sampleCount": len(samples),
               "maxSampledFootprintBytes": max((s["processFootprintBytes"] for s in samples), default=None),
               "minSampledAvailableBytes": min((s["availableBytes"] for s in samples), default=None),
               "maxPressureLevel": max((s["pressureLevel"] for s in samples), default=None),
               "maxSystemSwapGrowthBytes": max((s["systemSwapUsedBytes"] - before["systemSwapUsedBytes"]
                                                for s in samples), default=None),
               "reportDirectory": str(report_dirs[0]) if len(report_dirs) == 1 else None,
               "recordCount": len(records) if records is not None else None,
               "endUTC": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(args.output / "outcome.json", outcome)
    print(json.dumps(outcome, indent=2))
    if child.returncode != 0 or stop:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
