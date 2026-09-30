//
//  LeafAdmissionSourceShapeTests.swift
//  tesseractTests
//
//  Source-shape checks on the leaf producers (ADR-0078): only the Leaf
//  Admission builds a leaf Snapshot Admission, every producer reaches it,
//  and the MainActor closures around the manager's admit stay
//  non-suspending. The admission's behaviour is tested through its interface
//  in `LeafAdmissionTests`; these pin that no producer grows its own tail
//  again.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct LeafAdmissionSourceShapeTests {

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
    func onlyTheLeafAdmissionBuildsALeafSnapshotAdmission() throws {
        let admission = try readServerSource("LeafAdmission.swift")
        let producers = [
            ("the Leaf Store phase", try readLeafStorePhaseSources()),
            ("ServerCompletion.swift", try readServerSource("ServerCompletion.swift")),
            ("SpeculativePrefill.swift", try readServerSource("SpeculativePrefill.swift")),
        ]
        #expect(
            admission.components(separatedBy: "SnapshotAdmission.leaf(").count - 1 == 1,
            "The Leaf Admission builds the one leaf Snapshot Admission")
        for (name, source) in producers {
            #expect(
                !source.contains("SnapshotAdmission.leaf("),
                "\(name) must hand its cache to a Leaf Admission, not build the admission itself")
        }
    }

    @Test
    func everyLeafProducerReachesTheAdmission() throws {
        let leafPhase = try readLeafStorePhaseSources()
        let completion = try readServerSource("ServerCompletion.swift")
        let speculative = try readServerSource("SpeculativePrefill.swift")
        #expect(
            leafPhase.contains("private static func admitLeaf("),
            "The live, direct and boundary executors share one hand-over to the admission")
        #expect(
            leafPhase.components(separatedBy: "await admission.admit(").count - 1 == 1,
            "The executors' shared tail is their one call into the admission")
        #expect(
            leafPhase.components(separatedBy: "LeafAdmission.prepare(").count - 1 == 1,
            "Every executor prepares through the one context helper")
        #expect(
            speculative.components(separatedBy: "LeafAdmission.prepare(").count - 1 == 1
                && speculative.components(separatedBy: "await admission.admit(").count - 1 == 1,
            "The Speculative Canonical Prefill stores its leaf through one admission")
        #expect(
            completion.components(separatedBy: "LeafAdmission.prepare(").count - 1 == 1,
            "Salvage-on-cancel stores its leaf through one admission")
    }

    @Test
    func mainActorRunClosuresAroundPrefixCacheAdmissionsAreNonSuspending() throws {
        // The `MainActor.run` closures wrapping `prefixCache.admit` must stay
        // synchronous. `SSDSnapshotStore.tryEnqueue` is nonisolated under an
        // `NSLock`; an `await` inside the closure would force the HTTP hot
        // path to suspend mid-admission and break the ordering the
        // pending-ref map was designed around. There are exactly two: the
        // drive's mid-prefill checkpoint admission and the coalesced
        // admit-plus-stats hop inside the Leaf Admission (every leaf goes
        // through it).
        let source =
            try readServerSource("ServerCompletion.swift")
            + (try readLeafStorePhaseSources())
            + (try readServerSource("LeafAdmission.swift"))
        let bodies = extractMainActorRunBodies(
            source: source,
            containing: ["prefixCache.admit"]
        )
        #expect(
            bodies.count == 2,
            "Expected exactly 2 MainActor.run closures calling prefixCache admission APIs; found \(bodies.count). A refactor may have moved, collapsed, or duplicated the mid-prefill or Leaf Admission site; review before updating this assertion."
        )
        for (index, body) in bodies.enumerated() {
            #expect(
                !body.contains("await"),
                "MainActor.run closure #\(index) calling prefixCache admission APIs contains an `await`, so non-suspending admission is broken. Body:\n\(body)"
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
