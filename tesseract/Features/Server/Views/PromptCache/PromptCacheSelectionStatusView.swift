import SwiftUI

struct PromptCacheSelectionStatusView: View {
    let tree: PromptCacheTreeSnapshot?
    let node: PromptCacheTreeNodeSnapshot?
    var onShowDetails: (() -> Void)?

    var body: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Image(systemName: node == nil ? "point.3.connected.trianglepath.dotted" : "scope")
                .foregroundStyle(.secondary)

            if let node {
                Text("Offset \(node.tokenOffset)")
                    .font(.caption.weight(.semibold))
                Text(node.checkpointType ?? "path")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(node.storageState.displayName)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else if let tree {
                Text("\(tree.nodeCount) nodes")
                    .font(.caption.weight(.semibold))
                Text("\(tree.snapshotCount) snapshots")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(tree.partitionDigest)
                    .font(.caption.monospaced())
                    .foregroundStyle(.tertiary)
                    .truncationMode(.middle)
            } else {
                Text("No topology")
                    .font(.caption.weight(.semibold))
                Text("Waiting for prompt-cache snapshots")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 0)

            if let onShowDetails {
                Button {
                    onShowDetails()
                } label: {
                    Image(systemName: "info.circle")
                }
                .buttonStyle(.borderless)
                .help("Show selection details")
            }
        }
        .lineLimit(1)
        .padding(.horizontal, Theme.Spacing.sm)
        .padding(.vertical, 6)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: Theme.Radius.small, style: .continuous))
    }
}
