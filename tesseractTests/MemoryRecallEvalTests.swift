//
//  MemoryRecallEvalTests.swift
//  tesseractTests
//
//  Belief recall — the probe family the yardstick was missing (#332).
//
//  The episode eval (`MemoryEvalTests`) measures "bring back the right
//  conversation turns". Nothing measured "bring back the right *belief*" —
//  which is what the `recall` tool and the injection path actually rank — and
//  the gap hid a catastrophic embedding defect until the owner hit it live.
//
//  Probes are mechanical, from the live store's own structure:
//
//    sourceEcho — cue = the source episode a memory was distilled from;
//                 gold = every memory distilled from that episode. "The
//                 statement finds its distillate."
//    rareTerm   — cue = the memory's 2–3 rarest terms, as a short query;
//                 gold = every memory containing all of them. This is the
//                 shape that failed live ("Companion feature Tesseract").
//
//  Arms (same records, same pool — only the ranker differs):
//
//    search     — the REAL `MemoryEngine.search`, floor and all. What ships.
//    linear     — 0.75·cosine + 0.25·BM25, no floor (the shipped fusion).
//    rrf        — reciprocal-rank fusion, scale-free.
//    maxFusion  — max(linear, BM25): an exact keyword match can never lose.
//    noPrefix   — linear, cue embedded as a document (measures the instruct
//                 prefix's contribution).
//    noLayerNorm— linear, both sides pooled without the post-pooling
//                 layer-norm the HF reference does not have.
//    null       — cue-blind, newest-first. If anything scores near this,
//                 retrieval is decoration (#324's lesson).
//
//  Runs on a COPY of the owner's live store, read-only in spirit: nothing in
//  here ever touches the original. Gated on the store and the embedder
//  existing, so CI skips. Run it:
//
//      xcodebuild test -project tesseract.xcodeproj -scheme tesseract \
//        -destination 'platform=macOS' -parallel-testing-enabled NO \
//        -only-testing:tesseractTests/MemoryRecallEvalTests
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

// MARK: - Probes

nonisolated struct RecallProbe: Sendable {
    enum Family: String, CaseIterable, Sendable {
        case sourceEcho
        case rareTerm
    }

    let family: Family
    let cue: String
    let gold: Set<UUID>
}

// MARK: - The fixture

/// A throwaway copy of the owner's live memory store.
nonisolated enum RecallEvalStore {

    static var liveDirectory: URL? {
        let directory: URL
        if let override = ProcessInfo.processInfo.environment["TESSERACT_MEMORY_STORE"] {
            directory = URL(fileURLWithPath: NSString(string: override).expandingTildeInPath)
        } else {
            directory = MemoryEvalCorpus.applicationSupport.appendingPathComponent(
                "Tesseract Agent/agent/memory", isDirectory: true)
        }
        return FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("memory.sqlite").path)
            ? directory : nil
    }

    static var isAvailable: Bool {
        liveDirectory != nil && MemoryEvalCorpus.isEmbedderAvailable
    }

    struct Unavailable: Error {}

    /// Copy the store (plus WAL/SHM, mid-write) into a temp directory.
    static func copyToTemp() throws -> URL {
        guard let source = liveDirectory else { throw Unavailable() }
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("recall-eval-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: destination, withIntermediateDirectories: true)
        for suffix in ["", "-wal", "-shm"] {
            let file = source.appendingPathComponent("memory.sqlite\(suffix)")
            guard FileManager.default.fileExists(atPath: file.path) else { continue }
            try FileManager.default.copyItem(
                at: file,
                to: destination.appendingPathComponent("memory.sqlite\(suffix)"))
        }
        return destination
    }
}

// MARK: - Recall tool scope

/// The same-day blind spot (#332): a fact told this morning exists only as an
/// episode until sleep distills it. A beliefs-only recall cannot find it; the
/// tool's sweep must. Keyword-only on purpose — the same fail-open path a
/// fresh install takes — so this runs everywhere, embedder or not.
@MainActor
@Suite("Recall tool scope — the same-day blind spot", .serialized)
struct RecallScopeTests {

    @Test("A fact that exists only as an episode is still recallable")
    func sameDayEpisodeIsFound() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("recall-scope-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let store = try MemoryStore(directory: root)
        let engine = MemoryEngine(
            store: store,
            embedder: MemoryEmbedder(),
            isEnabled: { true },
            isDictationCaptureEnabled: { false },
            embedderDirectory: { nil }
        )
        try await store.append(
            Episode(
                source: .chat, conversationID: "c1", occurredAt: Date(),
                text: "I am working on the Tesseract Companion feature, "
                    + "a Jarvis-inspired personal assistant."))

        let (memories, episodes) = await engine.searchEverything(query: "Companion feature")
        #expect(memories.isEmpty, "no beliefs exist yet — sleep has not run")
        #expect(
            episodes.first?.episode.text.contains("Companion") == true,
            "the episode layer did not surface the same-day fact")
    }
}

// MARK: - Tool-truth smoke

/// The whole recall path below the model, through the REAL registered tool
/// (#332): `createRecallTool` → `searchEverything` → store + embedder →
/// formatted text. This is the seam-drop gate — ADR-0035 §12's first finding
/// was a feature whose every component tested green while the assembled path
/// silently returned nothing.
///
/// The one hop it cannot cover is the model choosing to call the tool and
/// reading its answer. That hop needs a second resident copy of the agent
/// model — deliberately not loaded from a test while the app may be running.
@MainActor
@Suite("Recall tool smoke — the tool the agent actually holds", .serialized)
struct RecallToolSmokeTests {

    @Test(
        "The registered recall tool surfaces the memory the query names",
        .enabled(if: MemoryEvalCorpus.isEmbedderAvailable))
    func recallToolFindsTheNamedMemory() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("recall-smoke-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let store = try MemoryStore(directory: root)
        let engine = MemoryEngine(
            store: store,
            embedder: MemoryEmbedder(),
            isEnabled: { true },
            isDictationCaptureEnabled: { false },
            embedderDirectory: { MemoryEvalCorpus.embedderDirectory }
        )
        await engine.prewarm()

        // The live failure, replayed through the real tool: the memory that
        // lost, and the distractors that beat it.
        await engine.remember(
            "The Companion feature is a core, Jarvis-inspired personal assistant "
                + "for the Tesseract application.")
        for distractor in [
            "I love cats.",
            "I use the Ghostty terminal emulator.",
            "He is from Europe.",
            "I like rain, especially summer rain.",
            "I exclusively use Celsius for temperature measurements.",
        ] {
            await engine.remember(distractor)
        }

        let tool = createRecallTool(memory: engine)
        let result = try await tool.execute(
            "recall-smoke-1", ["query": .string("Companion feature Tesseract")], nil, nil)
        let output = result.content.textContent

        let firstLine = try #require(output.split(separator: "\n").first.map(String.init))
        #expect(
            firstLine.contains("Companion"),
            "the tool's top line was \"\(firstLine)\", not the memory the query names")
    }
}

// MARK: - The suite

@MainActor
@Suite("Memory recall eval — belief retrieval on the live store", .serialized)
struct MemoryRecallEvalTests {

    // MARK: Probe generation

    private static func tokens(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 }
    }

    private static func makeProbes(
        memories: [MemoryRecord], episodeText: (UUID) async throws -> String?
    ) async throws -> [RecallProbe] {
        var probes: [RecallProbe] = []

        // sourceEcho — grouped by episode so shared sources yield one probe
        // with the full gold set.
        var goldByEpisode: [UUID: Set<UUID>] = [:]
        for memory in memories {
            for episodeID in memory.sourceEpisodeIDs {
                goldByEpisode[episodeID, default: []].insert(memory.id)
            }
        }
        for (episodeID, gold) in goldByEpisode {
            guard let text = try await episodeText(episodeID), !text.isEmpty else { continue }
            probes.append(RecallProbe(family: .sourceEcho, cue: text, gold: gold))
        }

        // rareTerm — document frequency over the memory texts themselves.
        var frequency: [String: Int] = [:]
        let tokenSets = memories.map { Set(tokens($0.text)) }
        for set in tokenSets {
            for token in set { frequency[token, default: 0] += 1 }
        }
        var seenCues = Set<String>()
        for memory in memories {
            let ordered = tokens(memory.text)
            var picked: [String] = []
            for token in ordered.sorted(by: {
                frequency[$0, default: 0] < frequency[$1, default: 0]
            })
            where picked.count < 3 && !picked.contains(token) {
                picked.append(token)
            }
            guard picked.count >= 2 else { continue }
            let cue = picked.joined(separator: " ")
            guard seenCues.insert(cue).inserted else { continue }
            let cueTerms = Set(picked)
            let gold = Set(
                zip(memories, tokenSets)
                    .filter { cueTerms.isSubset(of: $0.1) }
                    .map(\.0.id))
            probes.append(RecallProbe(family: .rareTerm, cue: cue, gold: gold))
        }
        return probes
    }

    // MARK: Ranking arms

    private enum Arm: String, CaseIterable {
        case search
        case linear
        case rrf
        case maxFusion
        case noPrefix
        case noLayerNorm
        case null
    }

    private struct ArmInputs {
        let cueVector: [Float]?
        let documentVectors: [UUID: [Float]]
        let keyword: [UUID: Double]
        let pool: [MemoryRecord]
    }

    /// Every arm but `search` (which is the engine itself) and `null` ranks
    /// with `MemoryEngine.relevance` or a declared *candidate* fusion of the
    /// same two signals. The candidate fusions are new math by definition —
    /// they are what the eval exists to judge.
    private static func rank(_ arm: Arm, _ inputs: ArmInputs, limit: Int = 10) -> [UUID] {
        switch arm {
        case .search:
            fatalError("the search arm goes through MemoryEngine.search")
        case .null:
            return inputs.pool.sorted { $0.bornAt > $1.bornAt }.prefix(limit).map(\.id)
        case .linear, .noPrefix, .noLayerNorm:
            return
                inputs.pool
                .map {
                    (
                        $0.id,
                        MemoryEngine.relevance(
                            cueVector: inputs.cueVector, id: $0.id,
                            vectorByID: inputs.documentVectors, keyword: inputs.keyword)
                    )
                }
                .sorted { $0.1 > $1.1 }.prefix(limit).map(\.0)
        case .rrf:
            let dense =
                inputs.pool
                .map { memory -> (UUID, Double) in
                    guard let cue = inputs.cueVector,
                        let vector = inputs.documentVectors[memory.id]
                    else { return (memory.id, 0) }
                    return (memory.id, MemoryStore.cosine(cue, vector))
                }
                .sorted { $0.1 > $1.1 }.map(\.0)
            let sparse = inputs.keyword.sorted { $0.value > $1.value }.map(\.key)
            var score: [UUID: Double] = [:]
            for (rank, id) in dense.enumerated() {
                score[id, default: 0] += 1.0 / Double(60 + rank + 1)
            }
            for (rank, id) in sparse.enumerated() {
                score[id, default: 0] += 1.0 / Double(60 + rank + 1)
            }
            return score.sorted { $0.value > $1.value }.prefix(limit).map(\.key)
        case .maxFusion:
            return
                inputs.pool
                .map { memory -> (UUID, Double) in
                    let linear = MemoryEngine.relevance(
                        cueVector: inputs.cueVector, id: memory.id,
                        vectorByID: inputs.documentVectors, keyword: inputs.keyword)
                    return (memory.id, max(linear, inputs.keyword[memory.id] ?? 0))
                }
                .sorted { $0.1 > $1.1 }.prefix(limit).map(\.0)
        }
    }

    // MARK: The original report

    @Test(
        "The query that failed live ranks its memory first",
        .enabled(if: RecallEvalStore.isAvailable))
    func theLiveFailureIsFixed() async throws {
        let directory = try RecallEvalStore.copyToTemp()
        defer { try? FileManager.default.removeItem(at: directory) }

        let store = try MemoryStore(directory: directory)
        let engine = MemoryEngine(
            store: store,
            embedder: MemoryEmbedder(),
            isEnabled: { true },
            isDictationCaptureEnabled: { false },
            embedderDirectory: { MemoryEvalCorpus.embedderDirectory }
        )
        await engine.prewarm()

        // Self-gating: the sentinel only fires while the owner's store still
        // holds a Companion memory. If it is ever deleted, there is nothing
        // to rank and nothing to assert.
        let memories = try await store.memories(status: nil, limit: 5_000)
        guard memories.contains(where: { $0.text.contains("Companion") }) else { return }

        let hits = await engine.search(query: "Companion feature Tesseract", limit: 10)
        try #require(!hits.isEmpty)
        #expect(
            hits.first?.memory.text.contains("Companion") == true,
            """
            2026-07-12's live failure is back: top hit for the Companion query \
            was "\(hits.first?.memory.text ?? "nil")"
            """)
    }

    // MARK: The eval

    @Test(
        "Belief recall: arms and judged calls on the owner's store",
        .enabled(if: RecallEvalStore.isAvailable))
    func beliefRecall() async throws {
        let directory = try RecallEvalStore.copyToTemp()
        defer { try? FileManager.default.removeItem(at: directory) }

        let store = try MemoryStore(directory: directory)
        let embedder = MemoryEmbedder()
        let engine = MemoryEngine(
            store: store,
            embedder: embedder,
            isEnabled: { true },
            isDictationCaptureEnabled: { false },
            embedderDirectory: { MemoryEvalCorpus.embedderDirectory }
        )
        // Prewarm reconciles the copied store onto the current embedding
        // scheme — the eval measures the vectors the app would actually have.
        await engine.prewarm()

        let memories = try await store.memories(status: nil, limit: 5_000)
        try #require(!memories.isEmpty, "the live store copy has no memories")
        let probes = try await Self.makeProbes(memories: memories) {
            try await store.episode(id: $0)?.text
        }
        try #require(!probes.isEmpty, "no probes could be derived from the store")

        // Vector variants. Documents: shipped scheme and the no-layer-norm
        // candidate. Cues: prefix × layer-norm.
        let texts = memories.map(\.text)
        let ids = memories.map(\.id)
        func documentVectors(layerNorm: Bool) async -> [UUID: [Float]] {
            var out: [UUID: [Float]] = [:]
            for start in stride(from: 0, to: texts.count, by: 32) {
                let slice = Array(texts[start..<min(start + 32, texts.count)])
                let vectors = await embedder.embed(slice, applyLayerNorm: layerNorm)
                guard vectors.count == slice.count else { continue }
                for (offset, vector) in vectors.enumerated() {
                    out[ids[start + offset]] = vector
                }
            }
            return out
        }
        let shippedDocs = await documentVectors(layerNorm: true)
        let rawDocs = await documentVectors(layerNorm: false)

        let cues = probes.map(\.cue)
        func cueVectors(prefix: Bool, layerNorm: Bool) async -> [String: [Float]] {
            var out: [String: [Float]] = [:]
            for start in stride(from: 0, to: cues.count, by: 32) {
                let slice = Array(cues[start..<min(start + 32, cues.count)])
                let vectors =
                    prefix
                    ? await embedder.embedQueries(slice, applyLayerNorm: layerNorm)
                    : await embedder.embed(slice, applyLayerNorm: layerNorm)
                guard vectors.count == slice.count else { continue }
                for (offset, vector) in vectors.enumerated() { out[slice[offset]] = vector }
            }
            return out
        }
        let shippedCues = await cueVectors(prefix: true, layerNorm: true)
        let bareCues = await cueVectors(prefix: false, layerNorm: true)
        let rawCues = await cueVectors(prefix: true, layerNorm: false)

        // Run every arm over every probe.
        var metrics: [RecallProbe.Family: [Arm: RetrievalMetrics]] = [:]
        var goldRelevances: [Double] = []
        var noiseRelevances: [Double] = []

        for probe in probes {
            let keyword = try await store.keywordScores(
                query: probe.cue, in: .memory, limit: 1_000)

            for arm in Arm.allCases {
                let ranked: [UUID]
                switch arm {
                case .search:
                    ranked = await engine.search(query: probe.cue, limit: 10, marksSeen: false)
                        .map(\.memory.id)
                case .noPrefix:
                    ranked = Self.rank(
                        arm,
                        ArmInputs(
                            cueVector: bareCues[probe.cue], documentVectors: shippedDocs,
                            keyword: keyword, pool: memories))
                case .noLayerNorm:
                    ranked = Self.rank(
                        arm,
                        ArmInputs(
                            cueVector: rawCues[probe.cue], documentVectors: rawDocs,
                            keyword: keyword, pool: memories))
                default:
                    ranked = Self.rank(
                        arm,
                        ArmInputs(
                            cueVector: shippedCues[probe.cue], documentVectors: shippedDocs,
                            keyword: keyword, pool: memories))
                }
                let hits = ranked.enumerated().filter { probe.gold.contains($0.element) }
                    .map(\.offset)
                metrics[probe.family, default: [:]][arm, default: RetrievalMetrics()].add(
                    rankedGoldHits: hits, goldCount: probe.gold.count, poolSize: memories.count)
            }

            // Floor evidence, under the shipped configuration.
            if let cueVector = shippedCues[probe.cue] {
                for memory in memories {
                    let relevance = MemoryEngine.relevance(
                        cueVector: cueVector, id: memory.id, vectorByID: shippedDocs,
                        keyword: keyword)
                    if probe.gold.contains(memory.id) {
                        goldRelevances.append(relevance)
                    } else {
                        noiseRelevances.append(relevance)
                    }
                }
            }
        }

        // The report.
        let report = MemoryEvalReport()
        report.section("Belief recall — \(memories.count) memories, \(probes.count) probes")
        for family in RecallProbe.Family.allCases {
            guard let byArm = metrics[family] else { continue }
            report.add("")
            report.add("  \(family.rawValue)")
            for arm in Arm.allCases {
                guard let m = byArm[arm]?.averaged else { continue }
                report.add(m.row(arm.rawValue))
            }
        }
        func percentile(_ values: [Double], _ p: Double) -> Double {
            guard !values.isEmpty else { return 0 }
            let sorted = values.sorted()
            return sorted[min(sorted.count - 1, Int(Double(sorted.count) * p))]
        }
        report.section("Relevance floor evidence (shipped configuration)")
        report.add(
            String(
                format: "  gold:  p10 %.3f  p50 %.3f  p90 %.3f",
                percentile(goldRelevances, 0.10), percentile(goldRelevances, 0.50),
                percentile(goldRelevances, 0.90)))
        report.add(
            String(
                format: "  noise: p50 %.3f  p90 %.3f  p99 %.3f  max %.3f",
                percentile(noiseRelevances, 0.50), percentile(noiseRelevances, 0.90),
                percentile(noiseRelevances, 0.99), noiseRelevances.max() ?? 0))
        report.add("  shipped floor: 0.2 — gold below it is lost, noise above it is shown")
        report.flush(name: "recall-eval")

        // The gates.
        for family in RecallProbe.Family.allCases {
            guard let byArm = metrics[family] else { continue }
            let shipped = byArm[.search]?.averaged.mrr ?? 0
            let null = byArm[.null]?.averaged.mrr ?? 0
            // Below the cue-blind null, retrieval is decoration (#324).
            #expect(
                shipped > null,
                "\(family.rawValue): shipped MRR \(shipped) does not beat cue-blind null \(null)")

            // "We shipped the winner": no judged variant may dominate the
            // shipped configuration by more than ε. When this fires, the right
            // response is to change the shipped configuration, not the gate.
            let epsilon = 0.05
            for challenger in [Arm.rrf, .maxFusion, .noPrefix, .noLayerNorm] {
                let mrr = byArm[challenger]?.averaged.mrr ?? 0
                #expect(
                    shipped >= mrr - epsilon,
                    "\(family.rawValue): \(challenger.rawValue) MRR \(mrr) dominates shipped \(shipped)"
                )
            }
        }
    }
}
