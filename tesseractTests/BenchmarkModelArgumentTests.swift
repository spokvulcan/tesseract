//
//  BenchmarkModelArgumentTests.swift
//  tesseractTests
//
//  The harnesses' two model arguments: `--bench-model` names a model folder,
//  `--bench-model-id` a catalog id. A catalog id given to the first fails at
//  once with the second's spelling, rather than as a missing folder deep in
//  the run.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct BenchmarkModelArgumentTests {

    private func failure(_ arguments: [String]) -> String? {
        let runner = BenchmarkRunner(config: .fromCommandLine(["Tesseract Agent"] + arguments))
        do {
            _ = try runner.resolveModelDirectory()
            return nil
        } catch {
            return error.localizedDescription
        }
    }

    @Test func aCatalogIDGivenAsAFolderNamesTheFlagItWasMeantFor() throws {
        let id = try #require(ModelDefinition.all.first?.id)
        #expect(
            failure(["--prefix-cache-e2e", "--bench-model", id])
                == "No model folder at \(URL(fileURLWithPath: id).path). --bench-model takes a "
                + "model folder: pass --bench-model-id \(id) for the catalog's.")
    }

    @Test func aMissingFolderIsNamed() {
        let folder = "/nonexistent/benchmark-model"
        #expect(
            failure(["--prefix-cache-e2e", "--bench-model", folder])
                == "No model folder at \(folder).")
    }
}
