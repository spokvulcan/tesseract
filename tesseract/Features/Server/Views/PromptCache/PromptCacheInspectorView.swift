import SwiftUI

struct PromptCacheInspectorView: View {
    let tree: PromptCacheTreeSnapshot?
    let node: PromptCacheTreeNodeSnapshot?
    let event: PromptCacheTelemetryEvent?

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            sectionHeader("Selection", symbol: "sidebar.right")

            if let node {
                nodeDetails(node)
            } else if let event {
                eventDetails(event)
            } else {
                ContentUnavailableView(
                    "No Selection",
                    systemImage: "scope",
                    description: Text("Select a tree node or event to inspect details.")
                )
                .frame(maxHeight: .infinity)
            }

            if let tree {
                Divider()
                sectionHeader("Partition", symbol: "square.stack.3d.up")
                detailRow("Digest", tree.partitionDigest)
                detailRow("Nodes", "\(tree.nodeCount)")
                detailRow("Snapshots", "\(tree.snapshotCount)")
                detailRow("Resident", PromptCacheFormatting.bytes(tree.totalSnapshotBytes))
                Text(tree.partitionSummary)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .textSelection(.enabled)
            }

            Spacer(minLength: 0)
        }
        .padding(Theme.Spacing.md)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: Theme.Radius.small, style: .continuous))
    }

    private func nodeDetails(_ node: PromptCacheTreeNodeSnapshot) -> some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            detailRow("Offset", "\(node.tokenOffset)")
            detailRow("Path", node.pathHash)
            detailRow("Depth", "\(node.depth)")
            detailRow("Edge tokens", "\(node.edgeTokenCount)")
            detailRow("Children", "\(node.childCount)")
            detailRow("State", node.storageState.displayName)
            detailRow("Checkpoint", node.checkpointType ?? "-")
            detailRow("RAM bytes", PromptCacheFormatting.bytes(node.snapshotBytes))
            detailRow("SSD bytes", PromptCacheFormatting.bytes(node.storageBytes))
            detailRow("Age", PromptCacheFormatting.age(node.lastAccessAgeSeconds))
            if let storageRefID = node.storageRefID {
                detailRow("Storage ref", storageRefID)
            }
            if let utility = node.utility {
                Divider()
                detailRow("Utility", String(format: "%.4f", utility))
                detailRow("Recency", node.normalizedRecency.map { String(format: "%.4f", $0) } ?? "-")
                detailRow("FLOP/byte", node.normalizedFlopEfficiency.map { String(format: "%.4f", $0) } ?? "-")
            }
        }
    }

    private func eventDetails(_ event: PromptCacheTelemetryEvent) -> some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            detailRow("Event", event.eventName)
            detailRow("Time", PromptCacheFormatting.timeFormatter.string(from: event.timestamp))
            detailRow("Scope", event.scope.rawValue)
            if let requestID = event.requestID {
                detailRow("Request", requestID.uuidString)
            }
            if let modelID = event.modelID {
                detailRow("Model", modelID)
            }
            ForEach(event.fields, id: \.self) { field in
                detailRow(field.key, field.value)
            }
        }
    }

    private func sectionHeader(_ title: String, symbol: String) -> some View {
        Label(title, systemImage: symbol)
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
    }

    private func detailRow(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.tertiary)
                .textCase(.uppercase)
            Text(value)
                .font(.caption.monospaced())
                .foregroundStyle(.primary)
                .lineLimit(2)
                .truncationMode(.middle)
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
