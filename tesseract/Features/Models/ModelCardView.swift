//
//  ModelCardView.swift
//  tesseract
//

import SwiftUI

struct ModelCardView: View {
    let model: ModelDefinition
    let status: ModelStatus
    let isMemoryLoaded: Bool
    let onDownload: () -> Void
    let onCancel: () -> Void
    let onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Top row: name + status badge
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.displayName)
                        .font(.headline)
                    Text(model.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                Spacer()

                statusBadge
            }

            // Size & memory info
            HStack(spacing: 12) {
                Label(model.sizeDescription, systemImage: "internaldrive")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if case .downloaded(let sizeOnDisk) = status {
                    Text("(\(ByteCountFormatter.string(fromByteCount: sizeOnDisk, countStyle: .file)) on disk)")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }

                Spacer()

                if isMemoryLoaded {
                    Label("Loaded", systemImage: "memorychip")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
            }

            // Progress bar
            if case .downloading(let progress) = status {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: progress)
                        .progressViewStyle(.linear)
                    Text("Downloading\u{2026} \(Int(progress * 100))%")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            // Action button
            HStack {
                if !model.dependencies.isEmpty, case .notDownloaded = status {
                    let depNames = model.dependencies.compactMap { depID in
                        ModelDefinition.all.first { $0.id == depID }?.displayName
                    }
                    if !depNames.isEmpty {
                        Text("Requires: \(depNames.joined(separator: ", "))")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }

                Spacer()

                actionButton
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var statusBadge: some View {
        switch status {
        case .downloaded:
            Label("Ready", systemImage: "checkmark.circle.fill")
                .font(.caption.weight(.medium))
                .foregroundStyle(.green)
        case .downloading:
            Label("Downloading", systemImage: "arrow.down.circle")
                .font(.caption.weight(.medium))
                .foregroundStyle(.orange)
                .symbolEffect(.pulse)
        case .notDownloaded:
            Label("Not Downloaded", systemImage: "arrow.down.to.line")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
        case .error(let msg):
            Label("Error", systemImage: "exclamationmark.triangle.fill")
                .font(.caption.weight(.medium))
                .foregroundStyle(.red)
                .help(msg)
        }
    }

    @ViewBuilder
    private var actionButton: some View {
        switch status {
        case .notDownloaded:
            Button {
                onDownload()
            } label: {
                Label("Download", systemImage: "arrow.down.circle")
            }
            .buttonStyle(.bordered)
        case .downloading:
            Button {
                onCancel()
            } label: {
                Label("Cancel", systemImage: "xmark.circle")
            }
            .buttonStyle(.bordered)
            .tint(.red)
        case .downloaded:
            Button(role: .destructive) {
                onDelete()
            } label: {
                Label("Delete", systemImage: "trash")
            }
            .buttonStyle(.bordered)
        case .error:
            Button {
                onDownload()
            } label: {
                Label("Retry", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.bordered)
        }
    }
}
