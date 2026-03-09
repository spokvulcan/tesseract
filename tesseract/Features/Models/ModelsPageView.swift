//
//  ModelsPageView.swift
//  tesseract
//

import SwiftUI

struct ModelsPageView: View {
    @EnvironmentObject private var container: DependencyContainer
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    var body: some View {
        Form {
            ForEach(ModelDefinition.byCategory(), id: \.0) { category, models in
                Section(category.rawValue) {
                    ForEach(models) { model in
                        ModelCardView(
                            model: model,
                            status: downloadManager.statuses[model.id] ?? .notDownloaded,
                            isMemoryLoaded: isModelLoadedInMemory(model),
                            onDownload: { downloadManager.download(modelID: model.id) },
                            onCancel: { downloadManager.cancelDownload(modelID: model.id) },
                            onDelete: { downloadManager.deleteModel(modelID: model.id) },
                            onVerify: { downloadManager.verifyAndRepair(modelID: model.id) }
                        )
                    }
                }
            }

            Section {
                StorageSummaryView(totalSize: downloadManager.totalCacheSize)
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Models")
        .onAppear {
            downloadManager.refreshAllStatuses()
        }
    }

    private func isModelLoadedInMemory(_ model: ModelDefinition) -> Bool {
        switch model.id {
        case "whisper-large-v3-turbo":
            return container.transcriptionEngine.isModelLoaded
        case "qwen3-tts-voicedesign":
            return container.speechEngine.isModelLoaded
        case "flux2-klein-4b":
            return container.imageGenEngine.isModelLoaded
        case "nanbeige4.1-3b":
            return container.agentEngine.isModelLoaded
        default:
            return false
        }
    }
}

// MARK: - Storage Summary

struct StorageSummaryView: View {
    let totalSize: Int64

    private let storageURL = ModelDownloadManager.modelStorageURL

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Storage: \(ByteCountFormatter.string(fromByteCount: totalSize, countStyle: .file))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(storageURL.path)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .textSelection(.enabled)
            }

            Spacer()

            Button("Show in Finder") {
                NSWorkspace.shared.open(storageURL)
            }
            .font(.caption)
            .buttonStyle(.bordered)
        }
    }
}
