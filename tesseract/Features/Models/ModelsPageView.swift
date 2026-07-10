//
//  ModelsPageView.swift
//  tesseract
//

import SwiftUI

/// The model library: every known model grouped by category in a native
/// grouped form (design language §4 — settings surfaces lean native, with
/// stock macOS metrics), presence on disk encoded in ink (solid =
/// downloaded, faded = not). Two layers: the form is content; the toolbar
/// and the floating glass action bar are the functional layer.
struct ModelsPageView: View {
    @EnvironmentObject private var container: DependencyContainer
    @EnvironmentObject private var downloadManager: ModelDownloadManager
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var selectedModelID: String?
    @State private var modelPendingDelete: ModelDefinition?

    var body: some View {
        Form {
            ForEach(ModelDefinition.byCategory(), id: \.0) { category, models in
                Section(category.rawValue) {
                    ForEach(models) { model in
                        ModelRowView(
                            model: model,
                            status: downloadManager.status(for: model.id),
                            isMemoryLoaded: isModelLoadedInMemory(model)
                        )
                        .contentShape(Rectangle())
                        .onTapGesture { selectedModelID = model.id }
                        .listRowBackground(
                            selectedModelID == model.id
                                ? Color.accentColor.opacity(0.12) : nil
                        )
                        .contextMenu { contextMenuItems(for: model) }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .frame(maxWidth: Theme.Layout.contentMaxWidth)
        .frame(maxWidth: .infinity)
        // The action bar floats over the scrolling form; the soft bottom
        // edge keeps row text legible as it passes under the glass.
        .scrollEdgeEffectStyle(.soft, for: .bottom)
        .safeAreaInset(edge: .bottom) {
            if let model = selectedModel {
                ModelsActionBar(
                    model: model,
                    status: downloadManager.status(for: model.id),
                    isMemoryLoaded: isModelLoadedInMemory(model),
                    onDownload: { downloadManager.download(modelID: model.id) },
                    onCancel: { downloadManager.cancelDownload(modelID: model.id) },
                    onVerify: { downloadManager.verifyAndRepair(modelID: model.id) },
                    onDelete: { modelPendingDelete = model }
                )
                .padding(.horizontal, Theme.Spacing.xxl)
                .padding(.vertical, Theme.Spacing.md)
            }
        }
        .animation(reduceMotion ? nil : .smooth(duration: 0.25), value: selectedModelID)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    NSWorkspace.shared.open(ModelDownloadManager.modelStorageURL)
                } label: {
                    Label("Show in Finder", systemImage: "folder")
                }
                .help("Open the model storage folder in Finder")
            }
        }
        .navigationTitle("Models")
        .navigationSubtitle(storageSubtitle)
        .confirmationDialog(
            deleteDialogTitle,
            isPresented: isConfirmingDelete,
            presenting: modelPendingDelete
        ) { model in
            Button("Delete", role: .destructive) {
                downloadManager.deleteModel(modelID: model.id)
            }
        } message: { model in
            Text(
                "Removes \(onDiskDescription(for: model)) from disk. "
                    + "You can download it again anytime."
            )
        }
        .onAppear {
            downloadManager.refreshAllStatuses()
            if selectedModelID == nil {
                selectedModelID = defaultSelection
            }
        }
    }

    // MARK: - Selection

    private var selectedModel: ModelDefinition? {
        selectedModelID.flatMap(ModelDefinition.withID)
    }

    /// The most relevant row to land on: what's running now, else what's
    /// on disk, else nothing (the bar stays hidden until a pick).
    private var defaultSelection: String? {
        ModelDefinition.all.first { isModelLoadedInMemory($0) }?.id
            ?? ModelDefinition.all.first { downloadManager.isDownloaded($0.id) }?.id
    }

    // MARK: - Row actions

    @ViewBuilder
    private func contextMenuItems(for model: ModelDefinition) -> some View {
        switch downloadManager.status(for: model.id) {
        case .notDownloaded:
            Button {
                downloadManager.download(modelID: model.id)
            } label: {
                Label("Download", systemImage: "arrow.down.circle")
            }
        case .downloading, .verifying:
            Button {
                downloadManager.cancelDownload(modelID: model.id)
            } label: {
                Label("Cancel", systemImage: "xmark.circle")
            }
        case .downloaded:
            Button {
                downloadManager.verifyAndRepair(modelID: model.id)
            } label: {
                Label("Verify Files", systemImage: "checkmark.shield")
            }
            if let path = downloadManager.modelPath(for: model.id) {
                Button {
                    NSWorkspace.shared.open(path)
                } label: {
                    Label("Show in Finder", systemImage: "folder")
                }
            }
            Button(role: .destructive) {
                modelPendingDelete = model
            } label: {
                Label("Delete", systemImage: "trash")
            }
        case .error:
            Button {
                downloadManager.download(modelID: model.id)
            } label: {
                Label("Retry Download", systemImage: "arrow.clockwise")
            }
        }
    }

    // MARK: - Delete confirmation

    private var isConfirmingDelete: Binding<Bool> {
        Binding(
            get: { modelPendingDelete != nil },
            set: { if !$0 { modelPendingDelete = nil } }
        )
    }

    private var deleteDialogTitle: String {
        "Delete \(modelPendingDelete?.displayName ?? "model")?"
    }

    private func onDiskDescription(for model: ModelDefinition) -> String {
        if case .downloaded(let sizeOnDisk) = downloadManager.status(for: model.id) {
            return ByteCountFormatter.string(fromByteCount: sizeOnDisk, countStyle: .file)
        }
        return model.sizeDescription
    }

    // MARK: - Storage

    private var storageSubtitle: String {
        let total = downloadManager.totalCacheSize
        guard total > 0 else { return "Nothing downloaded yet" }
        return "\(ByteCountFormatter.string(fromByteCount: total, countStyle: .file)) on disk"
    }

    // MARK: - Memory state

    private func isModelLoadedInMemory(_ model: ModelDefinition) -> Bool {
        switch model.category {
        case .speechToText:
            return model.id == container.settingsManager.selectedSpeechToTextModelID
                && container.transcriptionEngine.isModelLoaded
        case .textToSpeech:
            return model.id == "qwen3-tts-voicedesign" && container.speechEngine.isModelLoaded
        case .agent:
            return container.inferenceArbiter.loadedLLMModelID == model.id
        }
    }
}

// MARK: - Model row

/// Icon-light native form row: presence on disk is encoded in ink, not
/// badges — a downloaded model reads solid, an absent one faded. At most
/// one status glyph on the trailing edge; actions live in the context
/// menu and the action bar (design language §2).
private struct ModelRowView: View {
    let model: ModelDefinition
    let status: ModelStatus
    let isMemoryLoaded: Bool

    private var isOnDisk: Bool {
        if case .downloaded = status { return true }
        return false
    }

    var body: some View {
        HStack(spacing: Theme.Spacing.md) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName)
                    .fontWeight(.semibold)
                    .foregroundStyle(isOnDisk ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
                Text(model.description)
                    .font(.callout)
                    .foregroundStyle(
                        isOnDisk ? AnyShapeStyle(.secondary) : AnyShapeStyle(.tertiary)
                    )
                    .lineLimit(2)
            }

            Spacer(minLength: Theme.Spacing.md)

            trailingStatus
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private var trailingStatus: some View {
        HStack(spacing: Theme.Spacing.sm) {
            switch status {
            case .notDownloaded:
                Text(model.sizeDescription)
                    .foregroundStyle(.tertiary)
                    .monospacedDigit()
            case .downloading(let progress), .verifying(let progress):
                Text("\(Int(progress * 100)) %")
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                ProgressView(value: progress)
                    .progressViewStyle(.circular)
                    .controlSize(.small)
            case .downloaded(let sizeOnDisk):
                Text(ByteCountFormatter.string(fromByteCount: sizeOnDisk, countStyle: .file))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                if isMemoryLoaded {
                    Image(systemName: "memorychip")
                        .foregroundStyle(Color.accentColor)
                        .help("Loaded in memory")
                } else {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.secondary)
                        .help("Downloaded")
                }
            case .error(let message):
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .help(message)
            }
        }
    }
}
