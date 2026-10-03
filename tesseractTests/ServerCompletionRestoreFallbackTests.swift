import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

/// A planned restore whose snapshot fails to restore — a corrupt persisted
/// layer, or a Prefix-View Checkpoint whose Backing Leaf no longer fits it,
/// so `restore` throws and the restore is a cache miss — must run the whole
/// turn cold (ADR-0069 amendment), not only start from an empty cache. The
/// turn prefills the whole prompt from zero into a fresh cache, captures each
/// checkpoint where its rows really end, takes the cold image route when the
/// Cache Key Space carries an image, and admits nothing whose rows are not the
/// path it is stored under. The restore still reports `failedCopy`; the `lookup` event
/// and the `restored` phase add that the turn fell back.
///
/// Each case runs the real **Server Completion** module on the
/// content-relative toy **Model Session**, whose K/V row for a position is
/// the token id fed there, so a stored body can be read back against its
/// path.
@MainActor
struct ServerCompletionRestoreFallbackTests {
    typealias Replay = EmittedPathSynthesizedReplayTests

    private static let firstRequest = Replay.conversation([Replay.user("hi")])

    /// Under the preserve-thinking render a text turn captures no boundary
    /// helpers; under a think-stripping one it captures the last-user and
    /// last-message boundaries, which the warm plan would have labelled past
    /// the restore offset.
    @Test(arguments: [false, true])
    func aFailedRestoreRunsTheWholeTextTurnCold(stripsThinking: Bool) async throws {
        let context = stripsThinking ? TemplateRenderContext.canonical : Replay.preserving
        let fault = ToyRestoreFault()
        let session = Replay.Session(restoreFault: fault)
        let first = try await session.turn(
            Replay.conversation([Replay.user("hi")], context: context), context: context)
        // A text-only leaf is handed off without a restore; with the
        // check-out off it restores by copy, through the verb that fails.
        session.fixture.cacheAdmin.setLeafCheckoutDisabled(true)
        let request = Replay.conversation(
            [Replay.user("hi"), Replay.assistant("hello world"), Replay.user("more")],
            context: context)

        let (turn, verbs, captures) = try await Self.failedRestoreTurn(
            session, fault, request, text: "again", context: context)

        #expect(turn.text == "again")
        // The plan was a warm restore of what the first turn stored, by copy.
        let lookup = try #require(turn.event("lookup"))
        let planned = try #require(lookup.intField("snapshotOffset"))
        #expect(
            planned > 0 && planned <= first.render.count + first.fedGenerated.count, turn.account)
        #expect(lookup.field("restoreMode") == "failedCopy", turn.account)
        #expect(lookup.field("copyReason") == "checkoutDisabled", turn.account)
        // The turn ran cold: nothing skipped, the whole prompt fed from zero.
        #expect(lookup.field("restoreFallback") == "cold", turn.account)
        #expect(lookup.intField("skippedPrefillTokens") == 0, turn.account)
        #expect(lookup.intField("newTokensToPrefill") == turn.render.count, turn.account)
        #expect(turn.cached == 0, turn.account)
        #expect(turn.feeds.first?.position == 0, turn.account)
        #expect(turn.fedPrompt == turn.render, turn.account)
        Self.expectFreshCache(verbs)
        Self.expectRestoredPhaseFellBack(turn)
        if stripsThinking { #expect(!captures.isEmpty, turn.account) }
        Self.expectTrueOffsets(captures)
        try Self.expectResidentBodiesHoldTheirPaths(session)

        // The fault was one-shot: the next turn restores what this one stored.
        let next = try await session.turn(
            Replay.conversation(
                request.messages + [Replay.assistant("again"), Replay.user("next")],
                context: context),
            text: "done", context: context)
        #expect(next.text == "done")
        #expect(next.event("lookup")?.field("restoreFallback") == nil, next.account)
        #expect(next.event("lookup")?.field("restoreMode") == "copy", next.account)
        #expect(next.cached > planned, next.account)
        #expect(next.fedPrompt == Array(next.render[next.cached...]), next.account)
        try Self.expectResidentBodiesHoldTheirPaths(session)
    }

    /// A restore planned below a new image (ADR-0007 phase 2) continues
    /// through the image from the restored Position Anchor. When it fails,
    /// the turn takes the cold image route instead: the image prefix from
    /// zero through the anchored vision prepare, then the text tail.
    @Test func aFailedRestoreBelowAnImageTakesTheColdImageRoute() async throws {
        let imagePadID = EmittedPathToyTokenizer().imagePadID
        let runLength = EmittedPathToyTokenizer.imagePadRunLength
        let identity = ModelIdentity(
            configJSON: [
                "model_type": "qwen3_5", "image_token_id": imagePadID,
                "vision_config": ["num_heads": 16, "spatial_merge_size": 2],
            ],
            chatTemplate: nil)
        let fault = ToyRestoreFault()
        let session = Replay.Session(
            vision: ToyUserInputProcessor.VisionStub(
                padTokenId: imagePadID, padRunLength: runLength,
                frame: THW(1, 8, 8), expandsInPlace: true),
            identity: identity,
            restoreFault: fault)
        _ = try await session.turn(Self.firstRequest)
        let image = HTTPPrefixCacheImage(data: try Replay.tinyPNG())
        let request = Replay.conversation([
            Replay.user("hi"), Replay.assistant("hello world"),
            HTTPPrefixCacheMessage(role: .user, content: "look", images: [image]),
        ])

        let (turn, verbs, captures) = try await Self.failedRestoreTurn(
            session, fault, request, text: "nice")

        #expect(turn.text == "nice")
        // The processor's prompt: the template's one pad, expanded in place.
        let prompt = turn.render.flatMap { token in
            token == imagePadID ? Array(repeating: token, count: runLength) : [token]
        }
        let firstPad = try #require(prompt.firstIndex(of: imagePadID))
        // The plan was a restore below the image, by copy (an image key
        // space never hands off).
        let lookup = try #require(turn.event("lookup"))
        let planned = try #require(lookup.intField("snapshotOffset"))
        #expect(planned > 0 && planned <= firstPad, turn.account)
        #expect(lookup.field("restoreMode") == "failedCopy", turn.account)
        #expect(lookup.field("copyReason") == "imageKeySpace", turn.account)
        #expect(lookup.field("restoreFallback") == "cold", turn.account)
        #expect(lookup.intField("skippedPrefillTokens") == 0, turn.account)
        #expect(turn.cached == 0, turn.account)
        // The image prefix from zero, then the text tail: the whole prompt.
        #expect(turn.feeds.first?.position == 0, turn.account)
        #expect(turn.fedPrompt == prompt, turn.account)
        Self.expectFreshCache(verbs)
        #expect(verbs.contains(.visionContinuationQuery))
        Self.expectRestoredPhaseFellBack(turn)
        #expect(!captures.isEmpty, turn.account)
        Self.expectTrueOffsets(captures)
        try Self.expectResidentBodiesHoldTheirPaths(session, imagePadID: imagePadID)
    }

    /// The Speculation Plan is decided before the check-out and kept: a
    /// fallback turn does not ask again with nothing restored, so a resident
    /// MTP drafter, which only a turn that restores nothing engages, stays
    /// off. The drafter traps if it is engaged. A thinking-off turn under
    /// the preserve-thinking render predicts a direct leaf, the one leaf
    /// mode MTP accepts; the first turn samples at a temperature just above
    /// zero, which MTP refuses, so only the second could engage it.
    @Test func aFallbackTurnEngagesNoNewSpeculativeArm() async throws {
        let identity = ServerCompletionGenerationPromptTests.mtpEligibleIdentity
        let context = ServerCompletionGenerationPromptTests.context(
            enableThinking: false, preserveThinking: true)
        let fault = ToyRestoreFault()
        let session = Replay.Session(
            identity: identity, speculation: .inactiveMTP(pricedBy: identity),
            restoreFault: fault)
        var nearGreedy = ServerCompletionGenerationPromptTests.parameters()
        nearGreedy.temperature = 1e-4
        _ = try await session.turn(
            Replay.conversation([Replay.user("hi")], context: context),
            generated: session.completion(thinking: nil, text: "hello world"),
            context: context, parameters: nearGreedy)
        session.fixture.cacheAdmin.setLeafCheckoutDisabled(true)
        let request = Replay.conversation(
            [
                Replay.user("hi"), Replay.assistant("hello world", reasoning: ""),
                Replay.user("more"),
            ],
            context: context)

        await session.fixture.module.preemptSpeculativePrefill(on: session.fixture.actor)
        let verbsBefore = session.provider.recorder.verbs.count
        fault.arm()
        let turn = try await session.turn(
            request, generated: session.completion(thinking: nil, text: "again"),
            context: context)

        #expect(turn.text == "again")
        #expect(turn.leafStore["mode"] == HTTPLeafStoreMode.directLeaf.rawValue, turn.account)
        #expect(turn.event("lookup")?.field("restoreFallback") == "cold", turn.account)
        #expect(turn.fedPrompt == turn.render, turn.account)
        let verbs = Array(session.provider.recorder.verbs[verbsBefore...])
        #expect(!verbs.contains(.makeSpeculativeDecodeIterator), "\(verbs)")
    }

    // MARK: - Dropping the snapshot that failed

    /// The snapshot whose restore threw is dropped, so no later request pays
    /// the same failed restore. The fallback turn then plans its checkpoints
    /// again against the settled tree: the system checkpoint it lost is
    /// captured anew on the same turn, and the next request restores it.
    @Test func theSnapshotWhoseRestoreThrewIsDroppedAndRecaptured() async throws {
        let fault = ToyRestoreFault()
        let session = Replay.Session(restoreFault: fault)
        let first = try await session.turn(Self.firstRequest)
        let stable = try session.stablePrefix(of: first)

        let (turn, _, captures) = try await Self.failedRestoreTurn(
            session, fault, Replay.conversation([Replay.user("other")]), text: "again")

        #expect(turn.text == "again")
        let lookup = try #require(turn.event("lookup"))
        #expect(lookup.intField("snapshotOffset") == stable, turn.account)
        #expect(lookup.field("checkpointType") == "system", turn.account)
        #expect(lookup.field("restoreFallback") == "cold", turn.account)
        #expect(Self.dropEvent(turn)?.field("drop") == "body", turn.account)
        let failed = try #require(fault.failedBodyIDs.last)
        let resident = session.fixture.cacheAdmin.residentSnapshotsForTesting()
        #expect(!resident.contains { $0.snapshot.bodyID == failed })
        #expect(captures.contains(.init(label: stable, cacheOffset: stable)), "\(captures)")
        #expect(
            resident.contains { $0.snapshot.checkpointType == .system && $0.path.count == stable })
        try Self.expectResidentBodiesHoldTheirPaths(session)

        let next = try await session.turn(
            Replay.conversation([Replay.user("third")]), text: "done")
        #expect(next.text == "done")
        #expect(next.cached == stable, next.account)
        #expect(next.event("lookup")?.field("restoreFallback") == nil, next.account)
        #expect(next.fedPrompt == Array(next.render[stable...]), next.account)
    }

    /// A `RestoreError` says the layers themselves are corrupt, and the SSD
    /// copy holds the same bytes, so it goes with the RAM body; the fallback
    /// turn's fresh capture replaces it. Any other failure keeps the SSD
    /// copy, which the next hit hydrates. The system checkpoint is on disk
    /// only after the restart, so the failing turn is the one that hydrated
    /// it: the realistic way to meet a corrupt layer.
    @Test(arguments: [true, false])
    func corruptLayersDropTheSSDCopyToo(corrupt: Bool) async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("restore-fallback-ssd-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let ssdConfig = SSDPrefixCacheConfig(
            enabled: true, rootURL: root, budgetBytes: 1 << 30, maxPendingBytes: 1 << 30)
        let fault = ToyRestoreFault()
        let session = Replay.Session(ssdConfig: ssdConfig, restoreFault: fault)
        let first = try await session.turn(Self.firstRequest)
        let stable = try session.stablePrefix(of: first)
        await session.fixture.drain()
        await session.fixture.flush()
        let stored = try Self.manifestIDs(root, checkpointType: "system")
        #expect(stored.count == 1)
        session.restart(index: session.index, ssdConfig: ssdConfig)

        let error: any Error =
            corrupt
            ? HybridCacheSnapshot.RestoreError(layerIndex: 0, className: "KVCache", metaState: [])
            : ToyRestoreFault.Injected()
        let (turn, _, _) = try await Self.failedRestoreTurn(
            session, fault, Replay.conversation([Replay.user("other")]), text: "again",
            error: error)

        #expect(turn.text == "again")
        let lookup = try #require(turn.event("lookup"))
        #expect(lookup.field("hydratedFromSSD") == "true", turn.account)
        #expect(lookup.intField("snapshotOffset") == stable, turn.account)
        #expect(lookup.field("restoreFallback") == "cold", turn.account)
        #expect(
            Self.dropEvent(turn)?.field("drop") == (corrupt ? "bodyAndSSDCopy" : "body"),
            turn.account)
        await session.fixture.drain()
        await session.fixture.flush()
        let after = try Self.manifestIDs(root, checkpointType: "system")
        if corrupt {
            #expect(after.isDisjoint(with: stored), "\(after)")
            #expect(after.count == 1, "\(after)")
        } else {
            #expect(after == stored, "\(after)")
        }
    }

    // MARK: - Helpers

    /// The skip a fallback turn logs for the snapshot it dropped.
    private static func dropEvent(_ turn: Replay.Turn) -> PromptCacheTelemetryEvent? {
        turn.events.first {
            $0.eventName == "skip" && $0.field("stage") == "restore"
                && $0.field("reason") == "unrestorable-snapshot"
        }
    }

    /// The manifest's snapshot IDs of one checkpoint type.
    private static func manifestIDs(_ root: URL, checkpointType: String) throws -> Set<String> {
        let manifest = try JSONDecoder().decode(
            SnapshotManifest.self,
            from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
        return Set(manifest.snapshots.filter { $0.value.checkpointType == checkpointType }.keys)
    }

    /// Drive `request` with the restore fault armed; return the turn with
    /// the verbs and captures the toy session recorded during it.
    private static func failedRestoreTurn(
        _ session: Replay.Session, _ fault: ToyRestoreFault,
        _ request: HTTPPrefixCacheConversation, text: String,
        context: TemplateRenderContext = Replay.preserving,
        error: any Error = ToyRestoreFault.Injected()
    ) async throws -> (Replay.Turn, [ModelVerb], [ModelVerbRecorder.Capture]) {
        // A Speculative Canonical Prefill restores too; none may take the fault.
        await session.fixture.module.preemptSpeculativePrefill(on: session.fixture.actor)
        let recorder = session.provider.recorder
        let verbsBefore = recorder.verbs.count
        let capturesBefore = recorder.captures.count
        fault.arm(throwing: error)
        let turn = try await session.turn(request, text: text, context: context)
        return (
            turn, Array(recorder.verbs[verbsBefore...]),
            Array(recorder.captures[capturesBefore...])
        )
    }

    /// The restore was tried and failed, and the prefill ran on a new cache.
    private static func expectFreshCache(
        _ verbs: [ModelVerb], sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let restore = verbs.firstIndex(of: .restore)
        let newCache = verbs.firstIndex(of: .newCache)
        let prefill = verbs.firstIndex(of: .prefill)
        #expect(restore != nil, "\(verbs)", sourceLocation: sourceLocation)
        #expect(newCache != nil, "\(verbs)", sourceLocation: sourceLocation)
        if let restore, let newCache, let prefill {
            #expect(
                restore < newCache && newCache < prefill, "\(verbs)", sourceLocation: sourceLocation
            )
        }
    }

    /// The `restored` phase keeps `failedCopy` and says the turn fell back.
    private static func expectRestoredPhaseFellBack(
        _ turn: Replay.Turn, sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let restored = turn.events.first {
            $0.eventName == "requestMemory" && $0.field("phase") == "restored"
        }
        #expect(
            restored?.field("restoreMode") == "failedCopy", turn.account,
            sourceLocation: sourceLocation)
        #expect(
            restored?.field("restoreFallback") == "cold", turn.account,
            sourceLocation: sourceLocation)
    }

    /// Every checkpoint the turn captured is labelled with the offset its
    /// cache held.
    private static func expectTrueOffsets(
        _ captures: [ModelVerbRecorder.Capture],
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(
            captures.allSatisfy { $0.cacheOffset == $0.label }, "\(captures)",
            sourceLocation: sourceLocation)
    }

    /// Every body the cache holds is the rows of the path it is stored
    /// under: restored, it reaches the path's length, and its key column
    /// reads back the path's ids (an image run's pseudo-tokens read back as
    /// the pad that was fed there).
    private static func expectResidentBodiesHoldTheirPaths(
        _ session: Replay.Session, imagePadID: Int? = nil,
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let resident = session.fixture.cacheAdmin.residentSnapshotsForTesting()
        #expect(!resident.isEmpty, sourceLocation: sourceLocation)
        for (path, snapshot, backingLeaf) in resident {
            #expect(snapshot.tokenOffset == path.count, sourceLocation: sourceLocation)
            let restored = try snapshot.restore(backingLeaf: backingLeaf)
            let layer = try #require(restored.first, sourceLocation: sourceLocation)
            #expect(
                layer.offset == path.count,
                "\(snapshot.checkpointType) at \(path.count) holds \(layer.offset) rows",
                sourceLocation: sourceLocation)
            let keys = try #require(layer.state.first, sourceLocation: sourceLocation)
            let width = keys.dim(-1)
            let flat = keys.asType(.float32).asArray(Float.self)
            let rows = stride(from: 0, to: flat.count, by: width).map { Int(flat[$0]) }
            let expected = path.map { $0 < 0 ? imagePadID ?? $0 : $0 }
            #expect(
                rows == expected,
                "\(snapshot.checkpointType) at \(path.count) holds other rows",
                sourceLocation: sourceLocation)
        }
    }
}
