//
//  ZImageGenContentView.swift
//  tesseract
//

import SwiftUI
import UniformTypeIdentifiers

struct ZImageGenContentView: View {
    @EnvironmentObject private var zimageGenEngine: ZImageGenEngine
    @EnvironmentObject private var downloadManager: ModelDownloadManager
    @Environment(InferenceArbiter.self) private var arbiter

    @State private var prompt: String = ""
    @State private var negativePrompt: String = ""
    @State private var selectedSize: ImageSize = .large
    @State private var seedText: String = ""
    @State private var numSteps: Int = 50
    @State private var guidance: Float = 4.0
    @State private var generatedImage: NSImage?
    @State private var errorMessage: String?

    private var zimageModelID: String { "z-image" }
    private var isModelDownloaded: Bool {
        if case .downloaded = downloadManager.statuses[zimageModelID] {
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
            imageArea
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider()
            controlBar
                .padding(16)
        }
        .navigationTitle("Z-Image")
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
        } else if zimageGenEngine.isGenerating {
            VStack(spacing: 12) {
                ProgressView(
                    value: Double(zimageGenEngine.currentStep),
                    total: Double(zimageGenEngine.totalSteps)
                )
                .frame(width: 200)
                Text("Step \(zimageGenEngine.currentStep)/\(zimageGenEngine.totalSteps)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } else if zimageGenEngine.isLoading {
            VStack(spacing: 12) {
                ProgressView()
                    .controlSize(.large)
                Text(zimageGenEngine.loadingStatus.isEmpty ? "Loading model..." : zimageGenEngine.loadingStatus)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        } else if !isModelDownloaded {
            VStack(spacing: 8) {
                Image(systemName: "photo.artframe")
                    .font(.system(size: 48))
                    .foregroundStyle(.tertiary)
                Text("Download Z-Image from the Models page to get started")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        } else {
            VStack(spacing: 8) {
                Image(systemName: "photo.artframe")
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

            // Negative prompt row
            HStack(spacing: 12) {
                TextField("Negative prompt (optional)", text: $negativePrompt)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1)
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

                Stepper("Steps: \(numSteps)", value: $numSteps, in: 20...100)
                    .frame(width: 130)

                HStack(spacing: 4) {
                    Text("CFG:")
                    TextField("", value: $guidance, format: .number)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 50)
                }

                TextField("Seed", text: $seedText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)

                Button(action: generate) {
                    if zimageGenEngine.isGenerating {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Text("Generate")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!isModelDownloaded || zimageGenEngine.isGenerating || zimageGenEngine.isLoading || prompt.isEmpty)
            }
        }
    }

    // MARK: - Actions

    private func loadModelIfNeeded() async {
        guard isModelDownloaded,
              !zimageGenEngine.isModelLoaded,
              !zimageGenEngine.isLoading,
              let path = downloadManager.modelPath(for: zimageModelID)
        else { return }

        do {
            try await zimageGenEngine.loadModel(from: path)
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
        }
    }

    private func generate() {
        guard !prompt.isEmpty, isModelDownloaded else { return }
        errorMessage = nil
        let seed = UInt64(seedText) ?? 0
        let negPrompt = negativePrompt.trimmingCharacters(in: .whitespaces).isEmpty ? nil : negativePrompt

        Task {
            do {
                // Arbiter evicts co-resident LLM/TTS before granting exclusive GPU access
                generatedImage = try await arbiter.withExclusiveGPU(.imageGen) {
                    self.arbiter.releaseOtherImageEngine(keeping: .zImage)
                    if !self.zimageGenEngine.isModelLoaded {
                        await self.loadModelIfNeeded()
                        guard self.zimageGenEngine.isModelLoaded else { return nil }
                    }
                    return try await self.zimageGenEngine.generateImage(
                        prompt: prompt,
                        negativePrompt: negPrompt,
                        width: selectedSize.width,
                        height: selectedSize.height,
                        numSteps: numSteps,
                        guidance: guidance,
                        seed: seed
                    )
                }
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func saveImage(_ image: NSImage) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "zimage_generated.png"
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
