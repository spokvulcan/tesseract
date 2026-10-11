//
//  Harness.swift
//  tesseract
//
//  The **Harness** table: every headless run the app can make instead of the
//  owner's launch — the loaded-model benches, parity checks and trace replays
//  that `scripts/bench.sh`, `scripts/dev.sh` and their siblings start with a
//  flag. One row per harness: the flag, its label, where it keeps its
//  `latest.log`, and how it runs. The rows decide the Mac's **Launch Mode**
//  (`LaunchMode.current`) and run from `TesseractApp.init`.
//
//  A harness never runs the owner's background work and opens no window
//  (ADR-0095): it owns the GPU and the model it loads. None of the runners
//  needs anything `DependencyContainer.setup()` starts: each builds its own
//  engine and loads its model from the model folder.
//

import Foundation

nonisolated struct Harness: Sendable {
    /// The launch argument that selects it.
    let flag: String
    /// How its failure reads in the log.
    let label: String
    /// Where it keeps its `latest.log` under the benchmark output directory,
    /// the file the scripts tail; nil: the directory itself.
    let logSubdirectory: String?
    /// What a failed run exits with. `--benchmark` logs its failure and
    /// exits 0, as it always has: `scripts/bench.sh` reads the log.
    var failureExitCode: Int32 = 1
    /// The run itself; returning ends the process with 0.
    let run: @MainActor @Sendable () async throws -> Void

    /// Every harness, in precedence order: the first row whose flag the
    /// arguments carry runs. `--benchmark`, which `scripts/bench.sh` always
    /// passes, comes after `--paro-parity-bench` and the other harnesses
    /// `bench.sh` is asked for (`LaunchModeTests` lists them), so those win.
    static let table: [Harness] = [
        Harness(
            flag: "--paro-parity-bench", label: "PARO parity bench",
            logSubdirectory: "paro-parity-bench"
        ) {
            try await ParoParityBenchRunner(runner: BenchmarkRunner()).run()
        },
        Harness(
            flag: "--turboquant-bench", label: "TurboQuant bench",
            logSubdirectory: "turboquant-bench"
        ) {
            try await TurboQuantBenchRunner(runner: BenchmarkRunner()).run()
        },
        Harness(flag: "--ssd-read-bench", label: "SSD read bench", logSubdirectory: nil) {
            try await SSDReadBenchRunner(runner: BenchmarkRunner()).run()
        },
        Harness(
            flag: "--warm-parity-bench", label: "Warm Body parity bench",
            logSubdirectory: nil
        ) {
            try await WarmBodyParityBenchRunner(runner: BenchmarkRunner()).run()
        },
        Harness(
            flag: "--snapshot-bench", label: "Snapshot bench",
            logSubdirectory: "snapshot-bench"
        ) {
            try await SnapshotBenchRunner(runner: BenchmarkRunner()).run()
        },
        Harness(
            flag: "--prefix-detect-bench", label: "Prefix detect bench",
            logSubdirectory: "prefix-detect-bench"
        ) {
            try await PrefixDetectBenchRunner(runner: BenchmarkRunner()).run()
        },
        Harness(
            flag: "--tokenize-cache-bench", label: "Tokenize cache bench",
            logSubdirectory: "tokenize-cache-bench"
        ) {
            try await TokenizeCacheBenchRunner(runner: BenchmarkRunner()).run()
        },
        Harness(
            flag: "--agent-cpu-bench", label: "Agent CPU bench",
            logSubdirectory: "agent-cpu-bench"
        ) {
            try await AgentCpuBenchRunner(runner: BenchmarkRunner()).run()
        },
        Harness(flag: "--dflash2-bench", label: "DFlash2 bench", logSubdirectory: nil) {
            try await DFlash2BenchRunner(runner: BenchmarkRunner()).run()
        },
        Harness(
            flag: "--prefix-cache-e2e", label: "Prefix cache E2E",
            logSubdirectory: "prefix-cache-e2e"
        ) {
            try await PrefixCacheE2ERunner(runner: BenchmarkRunner()).run()
        },
        Harness(
            flag: "--benchmark", label: "Benchmark", logSubdirectory: nil,
            failureExitCode: 0
        ) {
            try await BenchmarkRunner().run()
        },
        Harness(
            flag: "--hybrid-cache-correctness", label: "Hybrid cache correctness",
            logSubdirectory: "hybrid-cache-correctness"
        ) {
            try await HybridCacheCorrectnessRunner(runner: BenchmarkRunner()).run()
        },
        Harness(
            flag: "--prefill-step-benchmark", label: "Prefill step benchmark",
            logSubdirectory: "prefill-step-benchmark"
        ) {
            try await PrefillStepBenchmarkRunner(runner: BenchmarkRunner()).run()
        },
        Harness(
            flag: "--paroquant-vlm-smoke", label: "ParoQuant VLM smoke",
            logSubdirectory: "paroquant-vlm-smoke"
        ) {
            try await ParoQuantVLMSmokeRunner(runner: BenchmarkRunner()).run()
        },
        Harness(
            flag: "--prepared-checkpoint-parity", label: "Prepared Checkpoint parity",
            logSubdirectory: "prepared-checkpoint-parity"
        ) {
            try await PreparedCheckpointParityRunner(runner: BenchmarkRunner()).run()
        },
        Harness(
            flag: "--rotated-checkpoint-parity", label: "Rotated Ternary Checkpoint parity",
            logSubdirectory: "rotated-checkpoint-parity"
        ) {
            try await RotatedCheckpointParityRunner(runner: BenchmarkRunner()).run()
        },
        Harness(flag: "--trace-replay", label: "Trace replay", logSubdirectory: "trace-replay") {
            try await TraceReplayRunner().run()
        },
    ]

    /// Runs the harness in a main-actor task, then ends the process: 0 once
    /// it returns, `failureExitCode` after logging a failure.
    @MainActor
    func start() {
        Task { @MainActor in
            do {
                try await run()
                exit(0)
            } catch {
                logFailure("\(label) failed: \(error)")
                exit(failureExitCode)
            }
        }
    }

    /// Logs a failure and appends it to the harness's `latest.log`, the file
    /// the scripts tail; a harness that failed before opening its log gets
    /// one.
    @MainActor
    private func logFailure(_ message: String) {
        Log.agent.error("\(message)")
        var directory = BenchmarkConfig.fromCommandLine().outputDir
        if let logSubdirectory { directory.append(path: logSubdirectory) }
        let url = directory.appending(path: "latest.log")
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: Data("[harness] \(message)\n".utf8))
    }
}

nonisolated extension LaunchMode {
    /// This process's mode, decided once from its arguments and environment
    /// against the harness table.
    static let current = decide(
        arguments: CommandLine.arguments,
        environment: ProcessInfo.processInfo.environment,
        harnessFlags: Harness.table.map(\.flag))

    /// The harness this process runs, if it is one.
    var harness: Harness? {
        guard case .harness(let flag) = self else { return nil }
        return Harness.table.first { $0.flag == flag }
    }
}
