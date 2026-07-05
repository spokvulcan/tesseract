import SwiftUI

/// The telemetry face of the Prompt Cache page: four large numbers —
/// token reuse, prefill time saved, RAM tier, SSD tier — a reuse-rate
/// sparkline, and one monospace meta line of diagnostics. Same
/// plain-content grammar as `ServerHeroBand`: no glass, cards, or tiles
/// in the content layer.
struct PromptCacheHeroBand: View {
    let snapshot: PromptCacheTelemetrySnapshot?
    let aggregate: PromptCacheTelemetryAggregate
    let samples: [PromptCacheMetricSample]
    let isLive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            // Degrade gracefully as the inspector / narrow windows eat
            // width: full row with sparkline → row without sparkline →
            // two-by-two grid. Numbers never wrap mid-value.
            ViewThatFits(in: .horizontal) {
                heroRow(spacing: Theme.Spacing.xxl * 1.5, includeSparkline: true)
                heroRow(spacing: Theme.Spacing.xl, includeSparkline: false)
                heroGrid
            }

            Text(metaLine)
                .font(.caption.monospaced())
                .foregroundStyle(.tertiary)
                .lineLimit(1)
                .truncationMode(.tail)
                .textSelection(.enabled)
        }
        .padding(.horizontal, Theme.Spacing.xl)
        .padding(.top, Theme.Spacing.lg)
        .padding(.bottom, Theme.Spacing.md)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Layout variants

    private func heroRow(spacing: CGFloat, includeSparkline: Bool) -> some View {
        HStack(alignment: .center, spacing: spacing) {
            reuseNumber
            savedNumber
            ramNumber
            ssdNumber

            if includeSparkline {
                Spacer(minLength: 0)

                RateSparkline(samples: samples.map(\.tokenReuseRate))
                    .frame(width: 148, height: 40)
            }
        }
    }

    private var heroGrid: some View {
        Grid(
            alignment: .leading,
            horizontalSpacing: Theme.Spacing.xxl,
            verticalSpacing: Theme.Spacing.md
        ) {
            GridRow {
                reuseNumber
                savedNumber
            }
            GridRow {
                ramNumber
                ssdNumber
            }
        }
    }

    private var reuseNumber: some View {
        HeroNumber(
            value: reuseValue.0,
            unit: reuseValue.1,
            label: "token reuse",
            isLive: isLive && aggregate.lookupCount > 0
        )
    }

    private var savedNumber: some View {
        HeroNumber(value: savedValue.0, unit: savedValue.1, label: "prefill saved", isLive: false)
    }

    private var ramNumber: some View {
        HeroNumber(value: ramValue.0, unit: ramValue.1, label: "ram tier", isLive: false)
    }

    private var ssdNumber: some View {
        HeroNumber(value: ssdValue.0, unit: ssdValue.1, label: ssdLabel, isLive: false)
    }

    // MARK: - Values

    private var reuseValue: (String, String?) {
        guard aggregate.promptTokens > 0 else { return ("—", nil) }
        return (String(format: "%.1f", aggregate.tokenReuseRate * 100), "%")
    }

    private var savedValue: (String, String?) {
        guard let seconds = snapshot?.counters.savedPrefillSeconds, seconds > 0 else {
            return ("—", nil)
        }
        if seconds < 90 { return (String(format: "%.0f", seconds), "s") }
        if seconds < 90 * 60 { return (String(format: "%.1f", seconds / 60), "min") }
        return (String(format: "%.1f", seconds / 3600), "h")
    }

    private var ramValue: (String, String?) {
        guard let snapshot else { return ("—", nil) }
        let used = PromptCacheFormatting.bytes(snapshot.residentSnapshotBytes)
        guard snapshot.memoryBudgetBytes > 0 else { return (used, nil) }
        return (used, "/ \(PromptCacheFormatting.bytes(snapshot.memoryBudgetBytes))")
    }

    private var ssdValue: (String, String?) {
        guard let ssd = snapshot?.ssd, ssd.enabled else { return ("—", nil) }
        let used = PromptCacheFormatting.bytes(ssd.currentBytes)
        guard ssd.budgetBytes > 0 else { return (used, nil) }
        return (used, "/ \(PromptCacheFormatting.bytes(ssd.budgetBytes))")
    }

    private var ssdLabel: String {
        guard let ssd = snapshot?.ssd else { return "ssd tier" }
        if !ssd.enabled { return "ssd tier · off" }
        // Free space, not policy, is the binding constraint — the PRD's
        // "nearly-full disk degrades to the floor and the panel says so".
        if ssd.budgetFloorBound { return "ssd tier · at floor, disk low" }
        if ssd.pendingCount > 0 { return "ssd tier · \(ssd.pendingCount) pending" }
        return "ssd tier"
    }

    private var metaLine: String {
        guard snapshot != nil || aggregate.lookupCount > 0 else {
            return "waiting for the first cache lookup"
        }
        var parts: [String] = []
        if aggregate.lookupCount > 0 {
            parts.append(
                String(format: "hit %.1f%%", aggregate.hitRate * 100)
                    + " (ram \(aggregate.hitCount) · ssd \(aggregate.ssdHitCount)"
                    + " · miss \(aggregate.missCount))")
        }
        if let snapshot {
            parts.append(pluralized(snapshot.snapshotCount, "snapshot"))
            parts.append(pluralized(snapshot.totalNodeCount, "node"))
            parts.append(pluralized(snapshot.partitionCount, "partition"))
            // The pressure band is invisible until pressure squeezes the
            // live budget below the load-time ceiling.
            if snapshot.budgetCeilingBytes > 0,
                snapshot.memoryBudgetBytes < snapshot.budgetCeilingBytes
            {
                parts.append(
                    "budget squeezed (ceiling "
                        + PromptCacheFormatting.bytes(snapshot.budgetCeilingBytes) + ")")
            }
            // Free-disk context next to the SSD gauge (PRD #150): only
            // when the dynamic budget actually measured the volume.
            if let freeDisk = snapshot.ssd.freeDiskBytes {
                parts.append("disk free " + PromptCacheFormatting.bytes(freeDisk))
            }
        }
        if aggregate.averageLookupMs > 0 {
            parts.append(
                "lookup " + PromptCacheFormatting.milliseconds(aggregate.averageLookupMs))
        }
        if aggregate.averageRestoreMs > 0 {
            parts.append(
                "restore " + PromptCacheFormatting.milliseconds(aggregate.averageRestoreMs))
        }
        if let tuner = snapshot?.tuner, tuner.phase != "unavailable" {
            parts.append(String(format: "α %.2f · %@", tuner.alpha, tunerPhaseLabel(tuner)))
        }
        return parts.joined(separator: " · ")
    }

    private func pluralized(_ count: Int, _ noun: String) -> String {
        "\(count) \(noun)\(count == 1 ? "" : "s")"
    }

    private func tunerPhaseLabel(_ tuner: PromptCacheTunerSnapshot) -> String {
        switch tuner.phase {
        case "waitingForFirstEviction": "lru"
        case "bootstrapping": "tuning \(tuner.bootstrapProgress)/\(tuner.bootstrapTarget)"
        case "tuned": "tuned"
        default: tuner.phase
        }
    }
}
