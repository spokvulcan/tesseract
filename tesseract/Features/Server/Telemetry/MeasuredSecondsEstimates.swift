//
//  MeasuredSecondsEstimates.swift
//  tesseract
//
//  The two device constants **Recovery Cost** is denominated in (PRD
//  #82, slice #84): prefill throughput (FLOPs/s) and SSD hydration
//  bandwidth (bytes/s). Pure EWMA folds over real operations — never
//  guessed constants once the first measurement lands. Carried by value
//  on the **Eviction Configuration**, exposed on the telemetry
//  snapshot, and stamped into every trace record so offline replay and
//  online scoring share the same units.
//

import Foundation

/// Rolling measured device estimates. A pure value: `recordingPrefill`
/// / `recordingHydration` return the folded copy, so every consumer —
/// and every test — crosses the same global-free seam.
nonisolated struct MeasuredSecondsEstimates: Codable, Sendable, Equatable {
    /// EWMA weight of a new sample. 0.2 damps single-operation noise
    /// (thermal spikes, cold page cache) while converging within a
    /// handful of operations.
    static let sampleWeight = 0.2

    /// Cold-start default: ~1 TFLOP/s effective prefill throughput, the
    /// right order of magnitude for 4-bit 4B-class models on Apple
    /// Silicon. Only ordering matters before the first measurement —
    /// both terminal-loss sites divide by the same value.
    static let defaultPrefillFlopsPerSecond = 1.0e12

    /// Cold-start default: ~1.5 GB/s sustained read, conservative for
    /// Apple SSDs.
    static let defaultHydrationBytesPerSecond = 1.5e9

    /// Smallest duration accepted as a real measurement. Sub-100 µs
    /// "operations" are timer noise, not throughput signal.
    static let minimumSampleSeconds = 1e-4

    private(set) var prefillFlopsPerSecond: Double
    private(set) var hydrationBytesPerSecond: Double
    private(set) var prefillSampleCount: Int
    private(set) var hydrationSampleCount: Int

    init(
        prefillFlopsPerSecond: Double = MeasuredSecondsEstimates.defaultPrefillFlopsPerSecond,
        hydrationBytesPerSecond: Double = MeasuredSecondsEstimates.defaultHydrationBytesPerSecond,
        prefillSampleCount: Int = 0,
        hydrationSampleCount: Int = 0
    ) {
        self.prefillFlopsPerSecond = prefillFlopsPerSecond
        self.hydrationBytesPerSecond = hydrationBytesPerSecond
        self.prefillSampleCount = prefillSampleCount
        self.hydrationSampleCount = hydrationSampleCount
    }

    /// Fold one observed prefill: `flops` of model work in `seconds`.
    /// The first real sample replaces the cold-start default outright;
    /// later samples blend at `sampleWeight`. Non-positive or sub-noise
    /// inputs return `self` unchanged.
    func recordingPrefill(flops: Double, seconds: Double) -> MeasuredSecondsEstimates {
        guard flops > 0, seconds >= Self.minimumSampleSeconds else { return self }
        var next = self
        next.prefillFlopsPerSecond = Self.fold(
            sample: flops / seconds,
            into: prefillFlopsPerSecond,
            sampleCount: prefillSampleCount
        )
        next.prefillSampleCount = prefillSampleCount + 1
        return next
    }

    /// Fold one observed SSD hydration: `bytes` read and composed in
    /// `seconds`. Same first-sample-replaces / later-samples-blend rule
    /// as `recordingPrefill`.
    func recordingHydration(bytes: Int, seconds: Double) -> MeasuredSecondsEstimates {
        guard bytes > 0, seconds >= Self.minimumSampleSeconds else { return self }
        var next = self
        next.hydrationBytesPerSecond = Self.fold(
            sample: Double(bytes) / seconds,
            into: hydrationBytesPerSecond,
            sampleCount: hydrationSampleCount
        )
        next.hydrationSampleCount = hydrationSampleCount + 1
        return next
    }

    /// The one EWMA rule: the first real sample replaces the cold-start
    /// default outright; later samples blend at `sampleWeight`.
    private static func fold(sample: Double, into current: Double, sampleCount: Int) -> Double {
        sampleCount == 0
            ? sample
            : current * (1 - sampleWeight) + sample * sampleWeight
    }
}
