#!/usr/bin/env python3
"""Run the #528 Warm Body parity gate under the owner manifest's resource bounds.

The manifest (`owner-plan.json`, committed before the run) is the whole plan:
this wrapper refuses anything that is not APPROVED, checks that the release
binary and model files on disk are the ones the manifest names, samples the
harness process every `sampleIntervalMilliseconds`, and terminates it on the
first breach of a stop threshold. A breached campaign is never retried here;
the owner re-derives the manifest and commits it again.
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

PAGE_COUNTERS = ("vm.page_free_count", "vm.page_speculative_count", "vm.page_purgeable_count")


def available_bytes(probe, page_size):
    pages = 0
    for name in PAGE_COUNTERS:
        pages += probe.sysctl(name, ctypes.c_uint32()).value
    return pages * page_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=pathlib.Path, required=True,
                        help="the Release binary inside Tesseract Agent.app")
    parser.add_argument("--plan", type=pathlib.Path, required=True, help="the committed owner-plan.json")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="a new results directory")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("status") != "APPROVED":
        parser.error("Manifest status is not APPROVED")
    for key in ("ownerApproval", "preRegistrationCommit", "appCommit", "vendorCommit",
                "instrumentationCommit", "releaseBinarySHA256", "privateOutputDirectory"):
        if not plan.get(key):
            parser.error(f"Manifest field {key} is empty: not executable")
    resources = plan["resources"]
    for key, value in resources.items():
        if value is None:
            parser.error(f"Manifest resource {key} is null: not executable")
    if args.output.exists():
        parser.error("Output must be a new directory")
    if subprocess.run(["pgrep", "-x", "Tesseract Agent"], capture_output=True).returncode == 0:
        parser.error("Quit the ordinary app before running the gate")
    binary_digest = file_digest(args.app)
    if binary_digest != plan["releaseBinarySHA256"]:
        parser.error("Release binary SHA-256 differs from the manifest")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest_committed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(args.plan)], capture_output=True).returncode == 0
    if not manifest_committed:
        parser.error("Commit the manifest before running")
    model_dir = pathlib.Path(plan["model"]["directory"])
    for name, key in (("config.json", "configSHA256"), ("tokenizer.json", "tokenizerSHA256"),
                      ("chat_template.jinja", "templateSHA256")):
        if file_digest(model_dir / name) != plan["model"][key]:
            parser.error(f"{name} SHA-256 differs from the manifest")

    args.output.mkdir(parents=True)
    args.output = args.output.resolve()
    model_files = []
    for path in sorted(model_dir.iterdir()):
        if path.is_file() and (path.suffix == ".safetensors" or path.suffix == ".json"
                               or path.suffix == ".jinja"):
            model_files.append({"modelFile": path.name, "bytes": path.stat().st_size,
                                "sha256": file_digest(path)})
    write_json(args.output / "model-files.json", model_files)

    probe = OSProbe()
    page_size = probe.sysctl("vm.pagesize", ctypes.c_uint32()).value
    before = probe.read(os.getpid())
    initial_available = available_bytes(probe, page_size)
    if initial_available < resources["minimumInitialAvailableBytes"]:
        parser.error(f"Initial available memory {initial_available} below the manifest minimum")
    if before["pressureLevel"] not in (1, 2):
        parser.error("Starting memory pressure must be normal or warning")
    disk_free = os.statvfs(str(args.output)).f_bavail * os.statvfs(str(args.output)).f_frsize

    command = [str(args.app.resolve()), "--warm-parity-bench", "--warm-parity-plan",
               str(args.plan.resolve()), "--bench-model", str(model_dir),
               "--bench-model-id", plan["model"]["id"], "--bench-output", str(args.output / "harness"),
               "--bench-source-revision", head, "-isServerEnabled", "NO"]
    write_json(args.output / "environment.json", {
        "plan": plan, "planSHA256": file_digest(args.plan), "command": command,
        "binarySHA256": binary_digest, "sourceRevision": head,
        "workingTreeDirty": subprocess.run(["git", "diff", "--quiet", "HEAD"],
                                           capture_output=True).returncode != 0,
        "os": subprocess.check_output(["sw_vers"], text=True),
        "physicalMemoryBytes": probe.sysctl("hw.memsize", ctypes.c_uint64()).value,
        "initialAvailableBytes": initial_available, "initialDiskFreeBytes": disk_free,
        "baselineSystemSwapUsedBytes": before["systemSwapUsedBytes"],
        "startUTC": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    for name in ("warm_body_parity.py", "allocation_inventory_probe.py"):
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
                samples.append(sample)
                swap_growth = sample["systemSwapUsedBytes"] - before["systemSwapUsedBytes"]
                if sample["pressureLevel"] >= resources["pressureStopLevel"] or sample["pressureLevel"] < 1:
                    stop = f"memory pressure level {sample['pressureLevel']}"
                elif sample["processFootprintBytes"] >= resources["footprintStopBytes"]:
                    stop = f"process footprint {sample['processFootprintBytes']}"
                elif sample["availableBytes"] < resources["minimumAvailableStopBytes"]:
                    stop = f"available memory {sample['availableBytes']}"
                elif swap_growth >= resources["swapGrowthStopBytes"]:
                    stop = f"system swap growth {swap_growth}"
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
    report_dirs = list((args.output / "harness").glob("warm-parity-*"))
    verdicts = None
    if len(report_dirs) == 1 and (report_dirs[0] / "verdicts.json").exists():
        verdicts = json.loads((report_dirs[0] / "verdicts.json").read_text())
    outcome = {"exitCode": child.returncode, "resourceStop": stop,
               "elapsedSeconds": time.monotonic() - started,
               "sampleCount": len(samples),
               "maxSampledFootprintBytes": max((s["processFootprintBytes"] for s in samples), default=None),
               "minSampledAvailableBytes": min((s["availableBytes"] for s in samples), default=None),
               "maxPressureLevel": max((s["pressureLevel"] for s in samples), default=None),
               "maxSystemSwapGrowthBytes": max((s["systemSwapUsedBytes"] - before["systemSwapUsedBytes"]
                                                for s in samples), default=None),
               "reportDirectory": str(report_dirs[0]) if len(report_dirs) == 1 else None,
               "verdicts": [{k: v[k] for k in ("caseID", "arm", "verdict")} for v in verdicts] if verdicts else None,
               "endUTC": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(args.output / "outcome.json", outcome)
    print(json.dumps(outcome, indent=2))
    if child.returncode != 0 or stop:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
