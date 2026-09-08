import Foundation
import MLX
import MLXLMCommon

/// Scalar-only request timeline. Sampling never evaluates an MLX graph or
/// retains a model, cache, snapshot, or tensor. Process measurements include
/// other co-resident work; sampled maxima are lower bounds, not allocator peaks.
nonisolated final class RequestMemoryTelemetry: @unchecked Sendable {
    enum Phase: String, Sendable {
        case preparing, restoring, restored, prefilling, dflashPreparing, prefilled
        case decoding, generationQuiescent, admittingCheckpoints, storingLeaf
        case capturingLeaf, preparingPayload, admittingLeaf
        case recordingRequest, finishingStream, releasingRequest, finished, settled
    }

    struct Sample: Sendable {
        var activeBytes: Int
        var cacheBytes: Int
        var lifetimePeakBytes: Int
        var footprintBytes: Int?
        var residentBytes: Int?
        var compressedBytes: Int?
        var systemSwapUsedBytes: UInt64?

        static func current() -> Self {
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(
                MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<integer_t>.stride)
            let result = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
                }
            }
            var swap = xsw_usage()
            var swapSize = MemoryLayout<xsw_usage>.size
            let swapResult = sysctlbyname("vm.swapusage", &swap, &swapSize, nil, 0)
            return Self(
                activeBytes: Memory.activeMemory, cacheBytes: Memory.cacheMemory,
                lifetimePeakBytes: Memory.peakMemory,
                footprintBytes: result == KERN_SUCCESS ? Int(clamping: info.phys_footprint) : nil,
                residentBytes: result == KERN_SUCCESS ? Int(clamping: info.resident_size) : nil,
                compressedBytes: result == KERN_SUCCESS ? Int(clamping: info.compressed) : nil,
                systemSwapUsedBytes: swapResult == 0 ? swap.xsu_used : nil)
        }
    }

    struct Event: PrefixCacheDiagnostics.Payload {
        let eventName = "requestMemory"
        let fields: [(String, String)]
    }

    private let context: PrefixCacheDiagnostics.Context
    private let sample: @Sendable () -> Sample
    private let lock = NSLock()
    private let started = ContinuousClock.now
    private var phaseStarted = ContinuousClock.now
    private var phase: Phase = .preparing
    private var sequence = 0
    private var sampledPeakActive = 0
    private var sampledPeakFootprint: Int?
    private var stopped = false
    private var facts: [String: String] = [:]

    init(
        context: PrefixCacheDiagnostics.Context,
        sample: @escaping @Sendable () -> Sample = { .current() }
    ) {
        self.context = context
        self.sample = sample
    }

    /// The returned task captures only this scalar recorder. The owner must
    /// call finish on every exit (including failed/cancelled starts).
    func startSampling() -> Task<Void, Never> {
        mark(.preparing)
        return Task.detached(priority: .utility) { [self] in
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(1)) } catch { return }
                guard tick() else { return }
            }
        }
    }

    @discardableResult
    func tick() -> Bool {
        lock.withLock {
            guard !stopped else { return false }
            emit(kind: "periodic")
            return true
        }
    }

    func mark(_ next: Phase, facts updates: [String: String] = [:]) {
        lock.withLock {
            guard !stopped else { return }
            // End sample attributes a peak during a synchronous operation to
            // the phase that just ran, before changing the phase label.
            let changed = next != phase
            if changed {
                emit(kind: "phaseEnd")
                phaseStarted = .now
            }
            phase = next
            facts.merge(updates) { _, new in new }
            if updates["requestCacheLayerCount"] != nil {
                facts["requestCacheMeasuredAtPhase"] = next.rawValue
            }
            if updates["treeSnapshotBytes"] != nil {
                facts["treeMeasuredAtPhase"] = next.rawValue
            }
            emit(kind: changed ? "phaseBegin" : "observation")
        }
    }

    func finish(outcome: String) {
        lock.withLock {
            guard !stopped else { return }
            let aborted = ["caller", "streamCancelled"].contains(facts["cancelSignalOrigin"] ?? "")
            facts["outcome"] =
                outcome == "completed" && aborted ? "cancelledDuringCleanup" : outcome
            emit(kind: "phaseEnd")
            phase = .finished
            phaseStarted = .now
            emit(kind: "terminal")
            stopped = true
        }
    }

    /// The shared driver also cancels on natural stream finish. Preserve the
    /// origin so that signal is not misreported as a user abort.
    func recordCancellationSignal(origin: String) {
        lock.withLock {
            guard !stopped, facts["cancelSignalOrigin"] == nil else { return }
            facts["cancelSignalOrigin"] = origin
            facts["cancelSignalElapsedMs"] = Self.milliseconds(started.duration(to: .now))
            emit(kind: "cancelSignal")
        }
    }

    /// One bounded observation after the driving task has returned. It may
    /// overlap a newer request; this is deliberately not labelled "idle".
    func sampleAfterRelease() async {
        do { try await Task.sleep(for: .seconds(1)) } catch { return }
        lock.withLock {
            phase = .settled
            phaseStarted = .now
            emit(kind: "afterRelease")
        }
    }

    private func emit(kind: String) {
        let reading = sample()
        if phase != .settled {
            sampledPeakActive = max(sampledPeakActive, reading.activeBytes)
            if let bytes = reading.footprintBytes {
                sampledPeakFootprint = max(sampledPeakFootprint ?? 0, bytes)
            }
        }
        sequence += 1
        let now = ContinuousClock.now
        var values = facts
        values.merge([
            "phase": phase.rawValue, "sampleKind": kind, "sequence": "\(sequence)",
            "elapsedMs": Self.milliseconds(started.duration(to: now)),
            "phaseElapsedMs": Self.milliseconds(phaseStarted.duration(to: now)),
            "activeMlxBytes": "\(reading.activeBytes)", "cachedMlxBytes": "\(reading.cacheBytes)",
            "processLifetimePeakMlxBytes": "\(reading.lifetimePeakBytes)",
            "sampledRequestPeakActiveMlxBytes": "\(sampledPeakActive)",
        ]) { _, new in new }
        if let bytes = reading.footprintBytes { values["processFootprintBytes"] = "\(bytes)" }
        if let bytes = sampledPeakFootprint {
            values["sampledRequestPeakFootprintBytes"] = "\(bytes)"
        }
        if let bytes = reading.residentBytes { values["processResidentBytes"] = "\(bytes)" }
        if let bytes = reading.compressedBytes { values["processCompressedBytes"] = "\(bytes)" }
        if let bytes = reading.systemSwapUsedBytes { values["systemSwapUsedBytes"] = "\(bytes)" }
        context.log(
            Event(fields: values.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }),
            level: kind == "periodic" ? .info : .notice)
    }

    private static func milliseconds(_ duration: Duration) -> String {
        let parts = duration.components
        return String(
            format: "%.3f", Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15)
    }

    /// Call only on the Model Session's thread while cache mutation is
    /// quiescent. innerState reads existing handles without slicing or eval.
    /// Logical array bytes can overlap views and exclude allocator padding.
    static func cacheFacts(_ cache: [any KVCache]) -> [String: String] {
        var attention = 0
        var recurrent = 0
        var other = 0
        for entry in cache {
            let bytes = entry.innerState().reduce(0) { $0 + $1.nbytes }
            if entry is MambaCache {
                recurrent += bytes
            } else if entry is KVCacheSimple || entry is RotatingKVCache
                || entry is QuantizedKVCache
            {
                attention += bytes
            } else {
                other += bytes
            }
        }
        return [
            "requestCacheAttentionArrayBytes": "\(attention)",
            "requestCacheRecurrentArrayBytes": "\(recurrent)",
            "requestCacheOtherArrayBytes": "\(other)",
            "requestCacheLayerCount": "\(cache.count)",
        ]
    }
}
