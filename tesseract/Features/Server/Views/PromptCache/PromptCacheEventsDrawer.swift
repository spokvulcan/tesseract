import SwiftUI

/// Slide-up event console for the Prompt Cache page (⌘`): every cache
/// event as one monospace log line with a colored verdict, full fields
/// under the selected line, and a status bar of lifetime cache vitals.
/// Same standard-material drawer grammar as `ServerConsoleDrawer`.
struct PromptCacheEventsDrawer: View {
    /// Newest events rendered; older ones stay reachable via search.
    /// The feed is a plain `VStack` (no lazy-stack re-entrancy traps),
    /// so it must not render the store's full 1 000-event buffer.
    private static let maxRenderedEvents = 200

    @Environment(PromptCacheTelemetryStore.self) private var telemetry

    let onClose: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            feed
            Divider()
            statusBar
        }
        .frame(height: 300)
        .background(.regularMaterial)
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Text("Events")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Spacer()
            if !telemetry.events.isEmpty {
                Button {
                    telemetry.clearEvents()
                } label: {
                    Image(systemName: "trash")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
                .help("Clear events and session stats")
            }
            Button(action: onClose) {
                Image(systemName: "xmark")
                    .font(.caption)
            }
            .buttonStyle(.borderless)
            .help("Close events (⌘`)")
        }
        .padding(.horizontal, Theme.Spacing.md)
        .padding(.vertical, 6)
    }

    // MARK: - Feed

    private var displayedEvents: [PromptCacheTelemetryEvent] {
        // `filteredEvents` is newest-first; the console reads oldest → newest.
        telemetry.filteredEvents.prefix(Self.maxRenderedEvents).reversed()
    }

    private var hiddenEventCount: Int {
        max(0, telemetry.filteredEvents.count - Self.maxRenderedEvents)
    }

    @ViewBuilder
    private var feed: some View {
        if telemetry.events.isEmpty {
            Text("No cache events yet")
                .font(.caption.monospaced())
                .foregroundStyle(.tertiary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    if hiddenEventCount > 0 {
                        Text("\(hiddenEventCount) older events hidden — search reaches them")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.quaternary)
                            .padding(.vertical, 2)
                            .padding(.horizontal, 6)
                    }
                    ForEach(displayedEvents) { event in
                        EventRow(
                            event: event,
                            isSelected: event.id == telemetry.selectedEventID,
                            onSelect: { toggleSelection(event) }
                        )
                        if event.id == telemetry.selectedEventID {
                            EventFieldLines(event: event)
                        }
                    }
                }
                .padding(.horizontal, Theme.Spacing.md)
                .padding(.vertical, Theme.Spacing.xs)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            // Follow new events once content overflows, but let a short
            // log sit at the top like a real console.
            .defaultScrollAnchor(.bottom, for: .sizeChanges)
        }
    }

    private func toggleSelection(_ event: PromptCacheTelemetryEvent) {
        telemetry.selectedEventID =
            telemetry.selectedEventID == event.id ? nil : event.id
    }

    // MARK: - Status bar

    private var statusBar: some View {
        HStack(spacing: Theme.Spacing.lg) {
            HStack(spacing: 5) {
                Circle()
                    .fill(telemetry.isLive ? .green : .orange)
                    .frame(width: 6, height: 6)
                Text(telemetry.isLive ? "live" : "paused")
            }

            if let saved = savedSummary {
                Text(saved)
                    .foregroundStyle(.green)
            }

            if telemetry.aggregate.evictionCount > 0 {
                Text("evict \(telemetry.aggregate.evictionCount)")
            }

            if telemetry.aggregate.rewindEventCount > 0 {
                Text(
                    "rewinds \(telemetry.aggregate.rewindEventCount) · ↩ "
                        + PromptCacheFormatting.compactNumber(telemetry.aggregate.rewindTokens)
                        + " tok"
                )
                .foregroundStyle(.orange)
            }

            Spacer()

            Text("\(telemetry.events.count) events")
                .foregroundStyle(.tertiary)
        }
        .font(.caption2.monospaced())
        .monospacedDigit()
        .foregroundStyle(.secondary)
        .padding(.horizontal, Theme.Spacing.md)
        .padding(.vertical, 5)
    }

    private var savedSummary: String? {
        guard let seconds = telemetry.snapshot?.counters.savedPrefillSeconds, seconds > 0 else {
            return nil
        }
        if seconds < 90 { return String(format: "saved %.0f s", seconds) }
        if seconds < 90 * 60 { return String(format: "saved %.1f min", seconds / 60) }
        return String(format: "saved %.1f h", seconds / 3600)
    }
}

// MARK: - Event row

private struct EventRow: View {
    let event: PromptCacheTelemetryEvent
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Text(isSelected ? "▾" : "▸")
                .foregroundStyle(.quaternary)
            Text(PromptCacheFormatting.timeFormatter.string(from: event.timestamp))
                .foregroundStyle(.tertiary)
            verdictText
            Text(summary)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer(minLength: 0)
        }
        .font(.caption.monospaced())
        .monospacedDigit()
        .padding(.vertical, 2.5)
        .padding(.horizontal, 6)
        .background(
            isSelected ? AnyShapeStyle(.quaternary.opacity(0.6)) : AnyShapeStyle(.clear),
            in: RoundedRectangle(cornerRadius: 5, style: .continuous)
        )
        .contentShape(Rectangle())
        .onTapGesture(perform: onSelect)
    }

    private var verdict: PromptCacheEventVerdict { PromptCacheEventVerdict(event: event) }

    private var verdictText: some View {
        Text(verdict.word)
            .foregroundStyle(verdict.color)
            .fontWeight(verdict.isEmphasized ? .semibold : .regular)
    }

    private var summary: String {
        var parts: [String] = []
        switch event.eventName {
        case "lookup":
            if let cached = event.intField("skippedPrefillTokens"),
                let prompt = event.intField("promptTokens")
            {
                parts.append("\(cached.formatted())/\(prompt.formatted()) tok")
            }
            if let type = event.field("checkpointType"), type != "nil" {
                parts.append(type)
            }
            if let lookup = event.doubleField("lookupMs") {
                parts.append("lookup " + PromptCacheFormatting.milliseconds(lookup))
            }
            if let restore = event.doubleField("restoreMs"), restore > 0 {
                parts.append("restore " + PromptCacheFormatting.milliseconds(restore))
            }
            if let shared = event.intField("sharedPrefixLength"),
                let floor = event.intField("snapshotOffset"),
                event.field("chainPrefixRestore") == "true"
                    || event.field("reason") == "chainPrefixHit",
                shared > floor
            {
                parts.append("↩ \((shared - floor).formatted()) tok")
            }

        case "ttft":
            if let ttft = event.doubleField("ttftMs") {
                parts.append(PromptCacheFormatting.milliseconds(ttft))
            }
            if let prefill = event.doubleField("prefillMs"), prefill > 0 {
                parts.append("prefill " + PromptCacheFormatting.milliseconds(prefill))
            }

        default:
            let reason = PromptCacheEventDisplay.reason(for: event)
            if reason != "-" { parts.append(reason) }
            let tokens = PromptCacheEventDisplay.tokenSummary(event)
            if tokens != "-" { parts.append(tokens) }
            let bytes = PromptCacheEventDisplay.bytesSummary(event)
            if bytes != "-" { parts.append(bytes) }
        }
        return parts.joined(separator: " · ")
    }
}

/// Console verdict for one telemetry event: the colored word at the
/// start of its log line.
private struct PromptCacheEventVerdict {
    let word: String
    let color: Color
    let isEmphasized: Bool

    init(event: PromptCacheTelemetryEvent) {
        switch event.eventName {
        case "lookup":
            let reason = event.field("reason") ?? ""
            let isHit = reason == "hit" || reason == "ssdHit" || reason == "chainPrefixHit"
            if !isHit {
                (word, color, isEmphasized) = ("miss", .orange, true)
            } else if event.field("hydratedFromSSD") == "true"
                || reason == "ssdHit" || reason == "chainPrefixHit"
            {
                (word, color, isEmphasized) = ("ssd", .cyan, true)
            } else {
                (word, color, isEmphasized) = ("hit", .green, true)
            }
        case "capture":
            (word, color, isEmphasized) = ("capture", .blue, false)
        case "eviction", "ssdEvictAtAdmission", "leafSupersession":
            (word, color, isEmphasized) = ("evict", .secondary, false)
        case "ssdAdmit":
            if let outcome = event.field("outcome"), outcome != "accepted" {
                (word, color, isEmphasized) = ("drop", .red, true)
            } else {
                (word, color, isEmphasized) = ("admit", .teal, false)
            }
        case "ttft":
            (word, color, isEmphasized) = ("ttft", .secondary, false)
        default:
            (word, color, isEmphasized) = (event.eventName, .secondary, false)
        }
    }
}

// MARK: - Event display helpers

/// Compact display strings for arbitrary telemetry events — shared by
/// the console rows and covered directly by store tests.
enum PromptCacheEventDisplay {
    static func symbol(for eventName: String) -> String {
        switch eventName {
        case "lookup": "magnifyingglass"
        case "capture": "tray.and.arrow.down"
        case "eviction", "ssdEvictAtAdmission": "trash"
        case "ssdAdmit": "internaldrive"
        case "ssdHit", "ssdRecordHit": "checkmark.circle"
        case "ssdMiss": "xmark.circle"
        case "memory": "memorychip"
        case "ttft": "timer"
        default: "circle"
        }
    }

    static func reason(for event: PromptCacheTelemetryEvent) -> String {
        event.field("reason") ?? event.field("outcome") ?? event.field("checkpointType") ?? "-"
    }

    static func tokenSummary(_ event: PromptCacheTelemetryEvent) -> String {
        if let cached = event.field("skippedPrefillTokens"),
            let prompt = event.field("promptTokens")
        {
            return "\(cached)/\(prompt)"
        }
        if let offset = event.field("offset") {
            return "@\(offset)"
        }
        return "-"
    }

    static func bytesSummary(_ event: PromptCacheTelemetryEvent) -> String {
        if let bytes = event.intField("bytes") {
            return PromptCacheFormatting.bytes(bytes)
        }
        if let bytes = event.intField("freedBytes") {
            return PromptCacheFormatting.bytes(bytes)
        }
        if let bytes = event.intField("totalSnapshotBytes") {
            return PromptCacheFormatting.bytes(bytes)
        }
        return "-"
    }

    static func requestSummary(_ event: PromptCacheTelemetryEvent) -> String {
        event.requestID.map { String($0.uuidString.prefix(8)) } ?? "system"
    }
}

// MARK: - Field lines

/// Every raw field of the selected event, console-style — the drawer's
/// answer to the old event table's detail columns.
private struct EventFieldLines: View {
    let event: PromptCacheTelemetryEvent

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            if let requestID = event.requestID {
                fieldLine(key: "request", value: String(requestID.uuidString.prefix(8)))
            }
            if let modelID = event.modelID {
                fieldLine(key: "model", value: modelID)
            }
            ForEach(event.fields, id: \.key) { field in
                fieldLine(key: field.key, value: field.value)
            }
        }
        .padding(.leading, 34)
        .padding(.bottom, 4)
    }

    private func fieldLine(key: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(key)
                .foregroundStyle(.secondary)
            Text(value)
                .foregroundStyle(.tertiary)
                .textSelection(.enabled)
        }
        .font(.caption2.monospaced())
        .monospacedDigit()
    }
}
