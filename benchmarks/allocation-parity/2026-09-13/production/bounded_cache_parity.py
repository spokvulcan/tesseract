#!/usr/bin/env python3
"""Run the fixed 2K production cache gate under the approved 48 GiB Mac bounds."""

import argparse
import ctypes
import datetime
import json
import os
import pathlib
import subprocess
import time

from allocation_inventory_probe import GIB, OSProbe, file_digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan = {"promptTokens": 2048, "leafOffset": 1024, "sentinelTokens": 1,
            "kvBits": None, "maxSeconds": 600, "sampleIntervalSeconds": 0.25,
            "footprintStopBytes": 32 * GIB, "swapGrowthStopBytes": 2 * GIB,
            "allowedPressureLevels": [1, 2], "model": "qwen3.8-27b",
            "speculativeDecodingExercised": False}
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    if args.output.exists():
        parser.error("Output must be a new directory")
    if subprocess.run(["pgrep", "-x", "Tesseract Agent"], capture_output=True).returncode == 0:
        parser.error("Quit the ordinary app before running the isolated gate")
    args.output.mkdir(parents=True)
    args.output = args.output.resolve()
    model_root = pathlib.Path.home() / "Library/Application Support/models"
    model_dir = model_root / "mlx-community_Qwen3.8-27B-4bit"
    model_files = []
    for name in (model_dir.name, "incoai_Qwen3.8-27B-DFlash2"):
        for path in sorted((model_root / name).iterdir()):
            if path.is_file() and (path.suffix == ".safetensors" or path.name in {
                    "config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"}):
                model_files.append({"modelFile": str(path.relative_to(model_root)),
                                    "bytes": path.stat().st_size, "sha256": file_digest(path)})
    write_json(args.output / "model-files.json", model_files)
    probe = OSProbe()
    before = probe.read(os.getpid())
    if before["pressureLevel"] != 1:
        parser.error("Starting memory pressure must be normal")
    command = [str(args.app.resolve()), "--hybrid-cache-correctness", "--bench-bounded-cache-parity",
               "--bench-model", str(model_dir), "--bench-model-id", plan["model"],
               "--bench-output", str(args.output / "harness"), "-isServerEnabled", "NO"]
    source = subprocess.check_output(["git", "diff", "--binary", "--", "tesseract", "tesseractTests"])
    (args.output / "source.patch").write_bytes(source)
    write_json(args.output / "environment.json", {
        "plan": plan, "command": command, "binarySHA256": file_digest(args.app),
        "sourceRevision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sourcePatchSHA256": file_digest(args.output / "source.patch"),
        "os": subprocess.check_output(["sw_vers"], text=True),
        "physicalMemoryBytes": probe.sysctl("hw.memsize", ctypes.c_uint64()).value,
        "baselineSystemSwapUsedBytes": before["systemSwapUsedBytes"],
        "startUTC": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    for name in ("bounded_cache_parity.py", "allocation_inventory_probe.py", "capture_memory_replay.py"):
        path = pathlib.Path(__file__).with_name(name)
        (args.output / name).write_bytes(path.read_bytes())
    diagnostic = pathlib.Path.home() / "Library/Application Support/CacheDiagnostics" / (
        datetime.datetime.now().astimezone().strftime("%Y-%m-%d") + ".jsonl")
    position = diagnostic.stat().st_size if diagnostic.exists() else 0
    inode = diagnostic.stat().st_ino if diagnostic.exists() else None
    samples = []
    stop = None
    started = time.monotonic()
    with (args.output / "app.log").open("wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "TESSERACT_ALLOCATION_DIAGNOSTICS": "1"})
        try:
            while child.poll() is None:
                try:
                    sample = probe.read(child.pid)
                except OSError:
                    if child.poll() is not None:
                        break
                    raise
                samples.append(sample)
                if sample["pressureLevel"] not in plan["allowedPressureLevels"]:
                    stop = "memory pressure"
                elif sample["processFootprintBytes"] >= plan["footprintStopBytes"]:
                    stop = "process footprint"
                elif sample["systemSwapUsedBytes"] - before["systemSwapUsedBytes"] >= plan["swapGrowthStopBytes"]:
                    stop = "system swap growth"
                elif time.monotonic() - started >= plan["maxSeconds"]:
                    stop = "deadline"
                if stop:
                    break
                time.sleep(plan["sampleIntervalSeconds"])
        except Exception as error:
            stop = str(error)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    events = []
    if diagnostic.exists():
        if inode is not None and diagnostic.stat().st_ino != inode or diagnostic.stat().st_size < position:
            stop = stop or "diagnostics rotated"
        else:
            with diagnostic.open("rb") as stream:
                stream.seek(position)
                for line in stream:
                    event = json.loads(line)
                    if event.get("eventName") in {"allocationMemory", "leafLeaseBegin", "leafLeaseEnd", "leafRewind"}:
                        events.append(event)
    write_json(args.output / "os-samples.json", samples)
    write_json(args.output / "allocation-events.json", events)
    reports = list((args.output / "harness/hybrid-cache-correctness").glob("correctness_*.json"))
    passed = len(reports) == 1 and json.loads(reports[0].read_text()).get("passed") is True
    outcome = {"exitCode": child.returncode, "resourceStop": stop, "passed": passed,
               "elapsedSeconds": time.monotonic() - started,
               "maxSampledFootprintBytes": max((s["processFootprintBytes"] for s in samples), default=None),
               "maxSystemSwapGrowthBytes": max((s["systemSwapUsedBytes"] - before["systemSwapUsedBytes"]
                                                for s in samples), default=None)}
    write_json(args.output / "outcome.json", outcome)
    print(json.dumps(outcome, indent=2))
    if child.returncode != 0 or stop or not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
