//
//  ModelsActionBar.swift
//  tesseract
//

import SwiftUI

/// Floating Liquid Glass action bar for the Models page — the sibling of
/// the Speech transport bar. Reflects the selected model and morphs its
/// primary action with download status; the capsule carries the quiet
/// status story (size, progress, in-memory, errors).
struct ModelsActionBar: View {
    let model: ModelDefinition
    let status: ModelStatus
    let isMemoryLoaded: Bool
    let onDownload: () -> Void
    let onCancel: () -> Void
    let onVerify: () -> Void
    let onDelete: () -> Void

    @Namespace private var glassNamespace
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Tall enough for the glass buttons and the status capsule alike, so
    /// the bar never resizes as elements come and go with status changes.
    private let controlRowHeight: CGFloat = 40

    var body: some View {
        VStack(spacing: Theme.Spacing.sm) {
            GlassEffectContainer(spacing: Theme.Spacing.md) {
                HStack(spacing: Theme.Spacing.md) {
                    primaryButton

                    if case .downloaded = status {
                        deleteButton
                    }

                    statusCapsule
                }
                .frame(height: controlRowHeight)
            }
            .animation(reduceMotion ? nil : .smooth(duration: 0.25), value: status)

            if let repoID = model.repoID {
                Text(repoID)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .textSelection(.enabled)
            }
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Controls

    @ViewBuilder
    private var primaryButton: some View {
        switch status {
        case .notDownloaded:
            Button(action: onDownload) {
                Label("Download", systemImage: "arrow.down")
                    .frame(minWidth: 90)
            }
            .buttonStyle(.glassProminent)
            .controlSize(.large)
            .help("Download \(model.displayName) (\(model.sizeDescription))")
            .glassEffectID("primary", in: glassNamespace)
        case .downloading, .verifying:
            Button(action: onCancel) {
                Label("Cancel", systemImage: "xmark")
                    .frame(minWidth: 90)
            }
            .buttonStyle(.glass)
            .controlSize(.large)
            .help("Cancel and keep the files fetched so far")
            .glassEffectID("primary", in: glassNamespace)
        case .downloaded:
            Button(action: onVerify) {
                Label("Verify", systemImage: "checkmark.shield")
                    .frame(minWidth: 90)
            }
            .buttonStyle(.glass)
            .controlSize(.large)
            .help("Check the files on disk and re-download anything damaged")
            .glassEffectID("primary", in: glassNamespace)
        case .error:
            Button(action: onDownload) {
                Label("Retry", systemImage: "arrow.clockwise")
                    .frame(minWidth: 90)
            }
            .buttonStyle(.glassProminent)
            .controlSize(.large)
            .help("Retry the download")
            .glassEffectID("primary", in: glassNamespace)
        }
    }

    private var deleteButton: some View {
        Button(role: .destructive, action: onDelete) {
            Label("Delete", systemImage: "trash")
                .labelStyle(.iconOnly)
        }
        .buttonStyle(.glass)
        .tint(.red)
        .controlSize(.large)
        .help("Delete the downloaded files from disk")
        .glassEffectID("delete", in: glassNamespace)
    }

    // MARK: - Status capsule

    private struct Status: Equatable {
        var text: String
        var isSpinning = false
        var isError = false
        var isInMemory = false
        var progress: Double?
    }

    private var barStatus: Status {
        switch status {
        case .notDownloaded:
            var text = "\(model.sizeDescription) download"
            let dependencyNames = model.dependencies.compactMap {
                ModelDefinition.withID($0)?.displayName
            }
            if !dependencyNames.isEmpty {
                text += " · requires \(dependencyNames.joined(separator: ", "))"
            }
            return Status(text: text)
        case .downloading(let progress):
            return Status(
                text: "Downloading… \(Int(progress * 100)) %",
                isSpinning: true,
                progress: progress
            )
        case .verifying(let progress):
            return Status(
                text: "Verifying… \(Int(progress * 100)) %",
                isSpinning: true,
                progress: progress
            )
        case .downloaded(let sizeOnDisk):
            let size = ByteCountFormatter.string(fromByteCount: sizeOnDisk, countStyle: .file)
            return Status(
                text: isMemoryLoaded ? "\(size) on disk · In memory" : "\(size) on disk",
                isInMemory: isMemoryLoaded
            )
        case .error(let message):
            return Status(text: message, isError: true)
        }
    }

    private var statusCapsule: some View {
        let status = barStatus
        return HStack(spacing: Theme.Spacing.sm) {
            if status.isSpinning {
                ProgressView()
                    .controlSize(.small)
            }
            if status.isInMemory {
                Image(systemName: "memorychip")
                    .foregroundStyle(Color.accentColor)
            }
            if status.isError {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }
            if let progress = status.progress {
                ProgressView(value: progress)
                    .frame(width: 56)
            }
            Text(status.text)
                .font(.callout)
                .foregroundStyle(status.isError ? .red : .primary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .padding(.horizontal, Theme.Spacing.lg)
        .padding(.vertical, Theme.Spacing.sm + 2)
        .glassEffect()
        .glassEffectID("status", in: glassNamespace)
    }
}
