//
//  ModelsPageView.swift
//  tesseract
//

import SwiftUI

struct ModelsPageView: View {
    @EnvironmentObject private var container: DependencyContainer
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                ForEach(ModelDefinition.byCategory(), id: \.0) { category, models in
                    VStack(alignment: .leading, spacing: 12) {
                        Label(category.rawValue, systemImage: category.symbolName)
                            .font(.title3.weight(.semibold))
                            .foregroundStyle(.primary)

                        ForEach(models) { model in
                            ModelCardView(
                                model: model,
                                status: downloadManager.statuses[model.id] ?? .notDownloaded,
                                isMemoryLoaded: isModelLoadedInMemory(model),
                                onDownload: { downloadManager.download(modelID: model.id) },
                                onCancel: { downloadManager.cancelDownload(modelID: model.id) },
                                onDelete: { downloadManager.deleteModel(modelID: model.id) }
                            )
                        }
                    }
                }

                CacheSummaryView(totalSize: downloadManager.totalCacheSize)
            }
            .padding(24)
        }
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
        default:
            return false
        }
    }
}

// MARK: - Cache Summary

struct CacheSummaryView: View {
    let totalSize: Int64

    private let cacheURL = URL.cachesDirectory.appendingPathComponent("mlx-audio")

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Cache: \(ByteCountFormatter.string(fromByteCount: totalSize, countStyle: .file))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(cacheURL.path)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .textSelection(.enabled)
            }

            Spacer()

            Button("Show in Finder") {
                NSWorkspace.shared.open(cacheURL)
            }
            .font(.caption)
            .buttonStyle(.bordered)
        }
        .padding(12)
        .background(
            .ultraThinMaterial,
            in: RoundedRectangle(cornerRadius: 12, style: .continuous)
        )
    }
}
