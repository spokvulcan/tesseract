//
//  ServerCompletionExtractSnapshotPayloadsTests.swift
//  tesseractTests
//
//  Source-shape checks on the leaf admission call sites: one owner builds
//  a leaf Snapshot Admission, and the MainActor closures around the
//  manager's admit stay non-suspending.
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

struct ServerCompletionExtractSnapshotPayloadsTests {

    // MARK: - Call-site wiring regression coverage
    //
    // The leaf Server Completion call sites cannot be exercised by unit tests
    // without a loaded MLX model — the full wiring is gated behind
    // `container.perform` on a real `ModelContainer`. These source
    // checks pin shared structured leaf capture and the synchronous
    // cache admission invariant.
    // Mirrors the `threadAffinityContractDocCommentIsPinned` pattern
    // at `HybridCacheSnapshotTests.swift:539`.

    private func readServerSource(_ fileName: String) throws -> String {
        let testFile = URL(fileURLWithPath: #filePath)
        let sourceFile =
            testFile
            .deletingLastPathComponent()  // tesseractTests
            .deletingLastPathComponent()  // project root
            .appendingPathComponent("tesseract")
            .appendingPathComponent("Features")
            .appendingPathComponent("Server")
            .appendingPathComponent(fileName)
        return try String(contentsOf: sourceFile, encoding: .utf8)
    }

    /// The **Leaf Store** phase is one type across several files (the
    /// phase, its executors, its report); the shape checks read all of them.
    private func readLeafStorePhaseSources() throws -> String {
        try [
            "LeafStorePhase.swift", "LeafStorePhase+Executors.swift",
            "LeafStorePhase+Report.swift",
        ]
        .map(readServerSource).joined(separator: "\n")
    }

    @Test
    func structuredLeafAdmissionStaysWithItsSingleOwner() throws {
        // Post ADR-0033: every Leaf Store executor (live, direct, boundary)
        // ends in the one shared `admitLeaf` tail, which — like the
        // speculative executor — routes through the one
        // `admitStructuredLeaf` owner, which alone constructs the leaf
        // Snapshot Admission value. The older dedicated
        // `strippedLeafPayload` path no longer exists under the single-leaf
        // policy.
        let leafPhase = try readLeafStorePhaseSources()
        let completion = try readServerSource("ServerCompletion.swift")
        let speculative = try readServerSource("SpeculativePrefill.swift")
        #expect(
            leafPhase.contains("private static func admitLeaf("),
            "The shared admit tail must exist so the live, direct and boundary executors share one leaf admission path"
        )
        #expect(
            completion.components(separatedBy: "SnapshotAdmission.leaf(").count - 1 == 1,
            "admitStructuredLeaf must be the only structured-leaf admission constructor in the completion module"
        )
        #expect(
            !leafPhase.contains("SnapshotAdmission.leaf(")
                && !speculative.contains("SnapshotAdmission.leaf("),
            "Leaf callers must route through admitStructuredLeaf, never construct admissions inline"
        )
        #expect(
            leafPhase.components(separatedBy: "await ServerCompletion.admitStructuredLeaf(").count
                - 1 == 1,
            "Every Leaf Store executor must reach the shared owner through the one admit tail"
        )
        #expect(
            speculative.components(separatedBy: "await ServerCompletion.admitStructuredLeaf(")
                .count - 1 == 1,
            "The speculative executor must route through the shared owner"
        )
        #expect(
            !completion.contains("leafPayload: strippedLeafPayload"),
            "Single-leaf policy should not retain the removed strippedLeafPayload store path"
        )
    }

    @Test
    func mainActorRunClosuresAroundPrefixCacheAdmissionsAreNonSuspending() throws {
        // The `MainActor.run` closures wrapping
        // `prefixCache.admit` must stay
        // synchronous. `SSDSnapshotStore.tryEnqueue` is nonisolated
        // under an `NSLock`; an `await` inside the closure would
        // force the HTTP hot path to suspend mid-admission and break
        // the ordering the pending-ref map was designed around.
        // Post ADR-0033 there are exactly two: the drive's mid-prefill
        // checkpoint admission and the coalesced admit-plus-stats hop
        // inside `admitStructuredLeaf` (every leaf path funnels there).
        let source =
            try readServerSource("ServerCompletion.swift")
            + (try readLeafStorePhaseSources())
        let bodies = extractMainActorRunBodies(
            source: source,
            containing: ["prefixCache.admit"]
        )
        #expect(
            bodies.count == 2,
            "Expected exactly 2 MainActor.run closures calling prefixCache admission APIs; found \(bodies.count). A refactor may have moved, collapsed, or duplicated the mid-prefill or admitStructuredLeaf site — review before updating this assertion."
        )
        for (index, body) in bodies.enumerated() {
            #expect(
                !body.contains("await"),
                "MainActor.run closure #\(index) calling prefixCache admission APIs contains an `await` — non-suspending admission is broken. Body:\n\(body)"
            )
        }
    }

    /// Scan `source` for every `MainActor.run { ... }` trailing-closure
    /// body and return the ones whose body contains at least one of
    /// `anchors`. Uses naive brace matching — adequate because the
    /// closures of interest (in `ServerCompletion.swift` and
    /// `LeafStorePhase.swift`) contain no string literals with braces,
    /// no block comments, and no nested closures wider than the
    /// enclosing `MainActor.run`. If that changes, the call-site count
    /// assertion above will start failing and the matcher can be
    /// upgraded then.
    private func extractMainActorRunBodies(
        source: String,
        containing anchors: [String]
    ) -> [String] {
        var result: [String] = []
        var cursor = source.startIndex
        let needle = "MainActor.run"
        while let runRange = source.range(of: needle, range: cursor..<source.endIndex) {
            guard let braceOpen = source[runRange.upperBound...].firstIndex(of: "{") else {
                break
            }
            var depth = 0
            var scanner = braceOpen
            var braceClose: String.Index?
            while scanner < source.endIndex {
                let ch = source[scanner]
                if ch == "{" {
                    depth += 1
                } else if ch == "}" {
                    depth -= 1
                    if depth == 0 {
                        braceClose = scanner
                        break
                    }
                }
                scanner = source.index(after: scanner)
            }
            guard let braceClose else { break }
            let bodyStart = source.index(after: braceOpen)
            let body = String(source[bodyStart..<braceClose])
            if anchors.contains(where: { body.contains($0) }) {
                result.append(body)
            }
            cursor = source.index(after: braceClose)
        }
        return result
    }
}
