import SwiftUI

struct PromptCacheEventTableView: View {
    @Bindable var store: PromptCacheTelemetryStore
    var onShowDetails: (() -> Void)?

    var body: some View {
        GeometryReader { proxy in
            VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
                header

                if proxy.size.width >= PromptCacheLayout.wideWidth {
                    eventTable
                } else {
                    compactEventList
                }
            }
            .padding(Theme.Spacing.sm)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: Theme.Radius.small, style: .continuous))
        }
    }

    private var header: some View {
        HStack {
            Label("Events", systemImage: "list.bullet.rectangle")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            Spacer(minLength: 0)

            if let selected = store.selectedEvent {
                Text(selected.eventName)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)

                if let onShowDetails {
                    Button {
                        onShowDetails()
                    } label: {
                        Image(systemName: "info.circle")
                    }
                    .buttonStyle(.borderless)
                    .help("Show event details")
                }
            }
        }
    }

    private var eventTable: some View {
        Table(store.filteredEvents, selection: $store.selectedEventID) {
            TableColumn("Time") { event in
                Text(PromptCacheFormatting.timeFormatter.string(from: event.timestamp))
                    .font(.caption.monospaced())
            }
            .width(72)

            TableColumn("Event") { event in
                Label(event.eventName, systemImage: PromptCacheEventDisplay.symbol(for: event.eventName))
                    .font(.caption)
            }
            .width(min: 120, ideal: 160)

            TableColumn("Reason") { event in
                Text(PromptCacheEventDisplay.reason(for: event))
                    .font(.caption.monospaced())
                    .lineLimit(1)
            }
            .width(min: 110, ideal: 150)

            TableColumn("Tokens") { event in
                Text(PromptCacheEventDisplay.tokenSummary(event))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            .width(min: 100, ideal: 130)

            TableColumn("Bytes") { event in
                Text(PromptCacheEventDisplay.bytesSummary(event))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            .width(min: 100, ideal: 130)

            TableColumn("Request") { event in
                Text(PromptCacheEventDisplay.requestSummary(event))
                    .font(.caption.monospaced())
                    .foregroundStyle(.tertiary)
            }
            .width(80)
        }
    }

    private var compactEventList: some View {
        List(store.filteredEvents, selection: $store.selectedEventID) { event in
            PromptCacheCompactEventRow(event: event)
                .tag(event.id)
        }
        .listStyle(.inset)
        .scrollContentBackground(.hidden)
    }
}

private struct PromptCacheCompactEventRow: View {
    let event: PromptCacheTelemetryEvent

    var body: some View {
        HStack(alignment: .top, spacing: Theme.Spacing.sm) {
            Image(systemName: PromptCacheEventDisplay.symbol(for: event.eventName))
                .foregroundStyle(.secondary)
                .frame(width: 16)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(event.eventName)
                        .font(.caption.weight(.semibold))
                        .lineLimit(1)
                    Text(PromptCacheFormatting.timeFormatter.string(from: event.timestamp))
                        .font(.caption2.monospaced())
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                    Spacer(minLength: 0)
                }

                HStack(spacing: Theme.Spacing.sm) {
                    Text(PromptCacheEventDisplay.reason(for: event))
                    Text(PromptCacheEventDisplay.tokenSummary(event))
                    Text(PromptCacheEventDisplay.bytesSummary(event))
                }
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
            }
        }
        .padding(.vertical, 3)
    }
}

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
           let prompt = event.field("promptTokens") {
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
