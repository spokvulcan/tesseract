//
//  ZImageGenEngine.swift
//  tesseract
//

import Foundation
import AppKit
import Combine
import os
import MLXImageGen

@MainActor
final class ZImageGenEngine: ObservableObject {
    @Published private(set) var isModelLoaded = false
    @Published private(set) var isLoading = false
    @Published private(set) var isGenerating = false
    @Published private(set) var loadingStatus: String = ""
    @Published private(set) var currentStep: Int = 0
    @Published private(set) var totalSteps: Int = 0

    private var pipeline: ZImagePipeline?

    func loadModel(from directory: URL) async throws {
        guard !isModelLoaded, !isLoading else { return }

        isLoading = true
        loadingStatus = "Loading Z-Image..."

        do {
            let actor = try await ZImagePipeline(modelDirectory: directory)
            pipeline = actor
            isModelLoaded = true
            loadingStatus = ""
            Log.image.info("Z-Image model loaded successfully")
        } catch {
            loadingStatus = ""
            Log.image.error("Failed to load Z-Image model: \(error)")
            throw error
        }

        isLoading = false
    }

    func releaseModel() {
        pipeline = nil
        isModelLoaded = false
        Log.image.info("Z-Image model released")
    }

    func generateImage(
        prompt: String,
        negativePrompt: String? = nil,
        width: Int = 1024,
        height: Int = 1024,
        numSteps: Int = 50,
        guidance: Float = 4.0,
        seed: UInt64 = 0
    ) async throws -> NSImage {
        guard let pipeline else {
            throw ZImageGenError.modelNotLoaded
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
            negativePrompt: negativePrompt,
            width: width,
            height: height,
            numSteps: numSteps,
            guidance: guidance,
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

enum ZImageGenError: LocalizedError {
    case modelNotLoaded

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: "Z-Image model is not loaded"
        }
    }
}
