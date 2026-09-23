#!/usr/bin/env python3
"""Run one loaded-model command under the #554 gate's approved stops.

    watchdog.py --output DIR -- <binary> <args...>

Samples the child every 250 ms and terminates it, without retry, when its
footprint passes 34 GiB, available memory falls under 6 GiB, system swap
grows by 1 GiB, or memory pressure reaches level 4 (or is unknown). The
stops are #553's resource block, which the owner approved for the gate.
"""
import argparse
import ctypes
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

SCRIPTS = pathlib.Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from allocation_inventory_probe import OSProbe, write_json  # noqa: E402
from warm_body_parity import available_bytes  # noqa: E402

RESOURCES = json.loads(
    (SCRIPTS.parent / "benchmarks/allocation-profile/2026-09-21/plan-compaction.json").read_text()
)["resources"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("No command to run")
    if args.output.exists():
        parser.error("Output must be a new directory")
    if subprocess.run(["pgrep", "-x", "Tesseract Agent"], capture_output=True).returncode == 0:
        parser.error("Quit the ordinary app first")
    args.output.mkdir(parents=True)
    probe = OSProbe()
    page_size = probe.sysctl("vm.pagesize", ctypes.c_uint32()).value
    before = probe.read(os.getpid())
    if before["pressureLevel"] != 1:
        parser.error("Starting memory pressure must be normal")
    initial_available = available_bytes(probe, page_size)
    if initial_available < RESOURCES["minimumInitialAvailableBytes"]:
        parser.error("Initial available memory below the plan minimum")
    write_json(args.output / "environment.json", {
        "command": command, "resources": RESOURCES, "initialAvailableBytes": initial_available,
        "baselineSystemSwapUsedBytes": before["systemSwapUsedBytes"],
        "startUTC": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    samples, stop = [], None
    started = time.monotonic()
    interval = RESOURCES["sampleIntervalMilliseconds"] / 1000
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
                if sample["pressureLevel"] >= RESOURCES["pressureStopLevel"] or sample["pressureLevel"] < 1:
                    stop = "memory pressure"
                elif sample["processFootprintBytes"] >= RESOURCES["footprintStopBytes"]:
                    stop = "process footprint"
                elif sample["availableBytes"] < RESOURCES["minimumAvailableStopBytes"]:
                    stop = "available memory"
                elif sample["systemSwapUsedBytes"] - before["systemSwapUsedBytes"] >= RESOURCES["swapGrowthStopBytes"]:
                    stop = "system swap growth"
                elif time.monotonic() - started >= RESOURCES["maximumCampaignSeconds"]:
                    stop = "deadline"
                if stop:
                    break
                time.sleep(interval)
        except Exception as error:  # a probe failure stops the run too
            stop = f"watchdog error: {error}"
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    write_json(args.output / "os-samples.json", samples)
    outcome = {
        "exitCode": child.returncode, "resourceStop": stop,
        "elapsedSeconds": time.monotonic() - started, "sampleCount": len(samples),
        "maxSampledFootprintBytes": max((s["processFootprintBytes"] for s in samples), default=None),
        "minSampledAvailableBytes": min((s["availableBytes"] for s in samples), default=None),
        "maxSystemSwapGrowthBytes": max(
            (s["systemSwapUsedBytes"] - before["systemSwapUsedBytes"] for s in samples), default=None),
        "maxPressureLevel": max((s["pressureLevel"] for s in samples), default=None),
        "endUTC": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(args.output / "outcome.json", outcome)
    print(json.dumps(outcome, indent=2))
    raise SystemExit(0 if child.returncode == 0 and not stop else 1)


if __name__ == "__main__":
    main()
