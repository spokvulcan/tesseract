//
//  ImageGenEngine.swift
//  tesseract
//

import Foundation
import AppKit
import Combine
import os
import MLXImageGen

@MainActor
final class ImageGenEngine: ObservableObject {
    @Published private(set) var isModelLoaded = false
    @Published private(set) var isLoading = false
    @Published private(set) var isGenerating = false
    @Published private(set) var loadingStatus: String = ""
    @Published private(set) var currentStep: Int = 0
    @Published private(set) var totalSteps: Int = 0

    private var pipeline: Flux2Pipeline?

    func loadModel(from directory: URL) async throws {
        guard !isModelLoaded, !isLoading else { return }

        isLoading = true
        loadingStatus = "Loading FLUX.2-klein-4B..."

        do {
            let actor = try await Flux2Pipeline(modelDirectory: directory)
            pipeline = actor
            isModelLoaded = true
            loadingStatus = ""
            Log.image.info("FLUX.2-klein-4B model loaded successfully")
        } catch {
            loadingStatus = ""
            Log.image.error("Failed to load image model: \(error)")
            throw error
        }

        isLoading = false
    }

    func releaseModel() {
        pipeline = nil
        isModelLoaded = false
        Log.image.info("Image generation model released")
    }

    func generateImage(
        prompt: String,
        width: Int = 1024,
        height: Int = 1024,
        numSteps: Int = 4,
        guidanceScale: Float = 3.5,
        seed: UInt64 = 0
    ) async throws -> NSImage {
        guard let pipeline else {
            throw ImageGenError.modelNotLoaded
        }

        isGenerating = true
        currentStep = 0
        totalSteps = numSteps

        defer {
            isGenerating = false
            currentStep = 0
            totalSteps = 0
        }

        let cgImage = try await pipeline.generateImage(
            prompt: prompt,
            width: width,
            height: height,
            numSteps: numSteps,
            guidanceScale: guidanceScale,
            seed: seed,
            onProgress: { [weak self] step, total in
                Task { @MainActor in
                    self?.currentStep = step
                    self?.totalSteps = total
                }
            }
        )

        return NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
    }
}

enum ImageGenError: LocalizedError {
    case modelNotLoaded

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: "Image generation model is not loaded"
        }
    }
}
