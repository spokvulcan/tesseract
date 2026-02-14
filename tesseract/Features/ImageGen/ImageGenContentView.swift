//
//  ImageGenContentView.swift
//  tesseract
//

import SwiftUI
import UniformTypeIdentifiers

struct ImageGenContentView: View {
    @ObservedObject var imageGenEngine: ImageGenEngine
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    @State private var prompt: String = ""
    @State private var selectedSize: ImageSize = .medium
    @State private var seedText: String = ""
    @State private var generatedImage: NSImage?
    @State private var errorMessage: String?

    private var fluxModelID: String { "flux2-klein-4b" }
    private var isModelDownloaded: Bool {
        if case .downloaded = downloadManager.statuses[fluxModelID] {
            return true
        }
        return false
    }

    private enum ImageSize: String, CaseIterable, Identifiable {
        case small = "512×512"
        case medium = "768×768"
        case large = "1024×1024"

        var id: String { rawValue }
        var width: Int {
            switch self {
            case .small: 512
            case .medium: 768
            case .large: 1024
            }
        }
        var height: Int { width }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Image display area
            imageArea
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            // Controls
            controlBar
                .padding(16)
        }
        .navigationTitle("Image")
    }

    // MARK: - Image Area

    @ViewBuilder
    private var imageArea: some View {
        if let image = generatedImage {
            Image(nsImage: image)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .padding(20)
                .contextMenu {
                    Button("Copy Image") {
                        let pb = NSPasteboard.general
                        pb.clearContents()
                        pb.writeObjects([image])
                    }
                    Button("Save Image...") {
                        saveImage(image)
                    }
                }
        } else if imageGenEngine.isGenerating {
            VStack(spacing: 12) {
                ProgressView(
                    value: Double(imageGenEngine.currentStep),
                    total: Double(imageGenEngine.totalSteps)
                )
                .frame(width: 200)
                Text("Step \(imageGenEngine.currentStep)/\(imageGenEngine.totalSteps)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } else if imageGenEngine.isLoading {
            VStack(spacing: 12) {
                ProgressView()
                    .controlSize(.large)
                Text(imageGenEngine.loadingStatus.isEmpty ? "Loading model..." : imageGenEngine.loadingStatus)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        } else if !isModelDownloaded {
            VStack(spacing: 8) {
                Image(systemName: "photo.fill")
                    .font(.system(size: 48))
                    .foregroundStyle(.tertiary)
                Text("Download FLUX.2-klein-4B from the Models page to get started")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        } else {
            VStack(spacing: 8) {
                Image(systemName: "photo.fill")
                    .font(.system(size: 48))
                    .foregroundStyle(.tertiary)
                Text("Enter a prompt and click Generate")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Controls

    private var controlBar: some View {
        VStack(spacing: 12) {
            if let errorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            HStack(spacing: 12) {
                TextField("Describe what you want to see...", text: $prompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...3)
                    .onSubmit { generate() }

                Picker("Size", selection: $selectedSize) {
                    ForEach(ImageSize.allCases) { size in
                        Text(size.rawValue).tag(size)
                    }
                }
                .frame(width: 120)

                TextField("Seed", text: $seedText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)

                Button(action: generate) {
                    if imageGenEngine.isGenerating {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Text("Generate")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!isModelDownloaded || imageGenEngine.isGenerating || imageGenEngine.isLoading || prompt.isEmpty)
            }
        }
    }

    // MARK: - Actions

    private func loadModelIfNeeded() async {
        guard isModelDownloaded,
              !imageGenEngine.isModelLoaded,
              !imageGenEngine.isLoading,
              let path = downloadManager.modelPath(for: fluxModelID)
        else { return }

        do {
            try await imageGenEngine.loadModel(from: path)
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
        }
    }

    private func generate() {
        guard !prompt.isEmpty, isModelDownloaded else { return }
        errorMessage = nil
        let seed = UInt64(seedText) ?? 0

        Task {
            do {
                // Load model on first generation
                if !imageGenEngine.isModelLoaded {
                    await loadModelIfNeeded()
                    guard imageGenEngine.isModelLoaded else { return }
                }
                generatedImage = try await imageGenEngine.generateImage(
                    prompt: prompt,
                    width: selectedSize.width,
                    height: selectedSize.height,
                    seed: seed
                )
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func saveImage(_ image: NSImage) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "generated_image.png"
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            if let tiffData = image.tiffRepresentation,
               let bitmap = NSBitmapImageRep(data: tiffData),
               let pngData = bitmap.representation(using: .png, properties: [:]) {
                try? pngData.write(to: url)
            }
        }
    }
}
