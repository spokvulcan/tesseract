//
//  WhisperModelContractTests.swift
//  tesseractTests
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct WhisperModelContractTests {

    private func makeTempModelDir(files: [String]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("whisper-contract-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for file in files {
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent(file), withIntermediateDirectories: true)
        }
        return dir
    }

    @Test
    func completeWhenBothCompiledModelsExist() throws {
        let dir = try makeTempModelDir(files: ["AudioEncoder.mlmodelc", "TextDecoder.mlmodelc"])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(WhisperModelContract.isComplete(at: dir))
        #expect(WhisperModelContract.missingFiles(at: dir).isEmpty)
    }

    @Test
    func reportsExactlyTheMissingFiles() throws {
        let dir = try makeTempModelDir(files: ["AudioEncoder.mlmodelc"])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(!WhisperModelContract.isComplete(at: dir))
        #expect(WhisperModelContract.missingFiles(at: dir) == ["TextDecoder.mlmodelc"])
    }

    @Test
    func emptyDirectoryMissesEverything() throws {
        let dir = try makeTempModelDir(files: [])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(WhisperModelContract.missingFiles(at: dir) == WhisperModelContract.requiredFiles)
    }
}
