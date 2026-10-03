import Foundation
import Testing

@testable import Tesseract_Agent

/// The **Speculation Plan**'s table (ADR-0079): which arm a request runs,
/// the allowance its rounds need and how its prefill splits, decided from
/// the request's facts alone. Both the Server Completion and the Raw
/// Generation Start read these rows; neither holds a rule of its own.
struct SpeculationPlanTests {

    /// The 27B profile: 24 attention heads, bf16 scores.
    private static let profile27B = ModelIdentity.FullAttentionScratchProfile(
        attentionHeads: 24, bytesPerElement: 2)

    private static func dflash2Drafter(contextWindow: Int = 16) -> ScriptedDFlash2Drafter {
        ScriptedDFlash2Drafter(for: ToyLanguageModel(script: [0]), contextWindow: contextWindow)
    }

    static let dflash2Only = Speculation(dflash2Drafter: dflash2Drafter())
    static let mtpOnly = Speculation(
        mtpDrafter: InactiveMTPDrafter(), scratchProfile: profile27B)
    static let both = Speculation(
        mtpDrafter: InactiveMTPDrafter(), dflash2Drafter: dflash2Drafter(),
        scratchProfile: profile27B)

    /// A greedy, text-only, unquantized, cold request of 2,048 tokens that
    /// stores a direct leaf: every arm's engaging case, so each test names
    /// only the fact it varies.
    private static func request(
        isTextOnly: Bool = true,
        kvBits: Int? = nil,
        kvScheme: KVScheme? = nil,
        temperature: Float = 0,
        promptTokens: Int = 2048,
        restoresPrefix: Bool = false,
        storedLeaf: HTTPLeafStoreMode? = .directLeaf
    ) -> SpeculationRequest {
        SpeculationRequest(
            isTextOnly: isTextOnly, kvBits: kvBits, kvScheme: kvScheme,
            temperature: temperature, promptTokens: promptTokens, restoresPrefix: restoresPrefix,
            storedLeaf: storedLeaf)
    }

    @Test func nothingResidentPlansNothing() {
        #expect(Speculation.none.plan(for: Self.request()) == nil)
    }

    // MARK: - Both arms

    @Test(arguments: [dflash2Only, mtpOnly, both])
    func imageBearingRequestsNeverSpeculate(_ speculation: Speculation) {
        #expect(speculation.plan(for: Self.request(isTextOnly: false)) == nil)
    }

    /// Speculation rewinds verify rows in place, which only unquantized KV
    /// supports, on either arm. The Raw Generation Start used to clear
    /// `kvBits` and speculate instead; nothing in the product sets it.
    @Test(arguments: [dflash2Only, mtpOnly, both], [.directLeaf, nil] as [HTTPLeafStoreMode?])
    func quantizedKVNeverSpeculates(_ speculation: Speculation, storedLeaf: HTTPLeafStoreMode?) {
        #expect(speculation.plan(for: Self.request(kvBits: 8, storedLeaf: storedLeaf)) == nil)
    }

    /// A TurboQuant KV Scheme keeps DFlash2: its verify writes and reads the
    /// compressed rows. MTP's head does not take one, so a scheme plans
    /// nothing where MTP is the only drafter.
    @Test(arguments: KVScheme.allCases)
    func aKVSchemeKeepsDFlash2AndRefusesMTP(_ scheme: KVScheme) throws {
        for speculation in [Self.dflash2Only, Self.both] {
            let plan = try #require(speculation.plan(for: Self.request(kvScheme: scheme)))
            #expect(plan.arm == .dflash2)
        }
        #expect(Self.mtpOnly.plan(for: Self.request(kvScheme: scheme)) == nil)
        #expect(Self.mtpOnly.plan(for: Self.request()) != nil)
    }

    // MARK: - DFlash2

    /// No cold, leaf-mode or greedy condition: the arm rides the keyed
    /// path's own restore and checkpoint-capturing prefill, so warm, tool,
    /// thinking and sampled turns engage, and so does the Raw Generation
    /// Start, which stores no leaf (ADR-0059).
    @Test(
        arguments: [false, true],
        [.directLeaf, .directToolLeaf, .canonicalUserLeaf, nil] as [HTTPLeafStoreMode?])
    func dflash2EngagesOnEveryTextOnlyUnquantizedRequest(
        restoresPrefix: Bool, storedLeaf: HTTPLeafStoreMode?
    ) throws {
        for temperature: Float in [0, 0.6] {
            let plan = try #require(
                Self.dflash2Only.plan(
                    for: Self.request(
                        temperature: temperature, promptTokens: 200_000,
                        restoresPrefix: restoresPrefix, storedLeaf: storedLeaf)))
            #expect(plan.arm == .dflash2)
            #expect(plan.advanceAllowance == DFlash2Support.blockSize)
            #expect(!plan.prefillsWholePrompt)
        }
    }

    @Test func dflash2IsPreferredWhenBothDraftersAreResident() {
        #expect(Self.both.plan(for: Self.request())?.arm == .dflash2)
    }

    /// Past the deepest planned capture, so every checkpoint and boundary
    /// snapshot lands before the iterator takes over.
    @Test func dflash2SplitsPastTheDeepestCapture() throws {
        let plan = try #require(Self.dflash2Only.plan(for: Self.request()))
        #expect(
            plan.prefillSplit(
                checkpointOffsets: [10, 40], executionBaseOffset: 0, promptTokens: 50) == 40)
    }

    /// A deep prompt keeps the pipelined driver for its bulk: the iterator
    /// prefills only its context window (16 rows here).
    @Test func dflash2SplitStartsNoEarlierThanItsWindow() throws {
        let plan = try #require(Self.dflash2Only.plan(for: Self.request()))
        #expect(
            plan.prefillSplit(
                checkpointOffsets: [10], executionBaseOffset: 0, promptTokens: 100) == 83)
    }

    /// The iterator samples the first token from the final prompt position,
    /// so a capture at the prompt's end still leaves it that one token.
    @Test func dflash2SplitLeavesTheFinalTokenToTheIterator() throws {
        let plan = try #require(Self.dflash2Only.plan(for: Self.request()))
        #expect(
            plan.prefillSplit(
                checkpointOffsets: [50], executionBaseOffset: 0, promptTokens: 50) == 49)
    }

    @Test func dflash2SplitStartsAtTheRestoredOffset() throws {
        let plan = try #require(Self.dflash2Only.plan(for: Self.request(restoresPrefix: true)))
        #expect(
            plan.prefillSplit(
                checkpointOffsets: [], executionBaseOffset: 30, promptTokens: 40) == 30)
    }

    // MARK: - MTP

    @Test func mtpEngagesOnAGreedyColdDirectLeafTurnWithinTheBudget() throws {
        let plan = try #require(Self.mtpOnly.plan(for: Self.request()))
        #expect(plan.arm == .mtp)
        #expect(plan.advanceAllowance == MTPDrafterSupport.blockSize)
        #expect(plan.prefillsWholePrompt)
        // The iterator takes every prompt row: nothing is left to the driver.
        #expect(
            plan.prefillSplit(
                checkpointOffsets: [10, 40], executionBaseOffset: 0, promptTokens: 50)
                == 0)
    }

    /// The Qwen heads are greedy-only; engaging at temperature > 0 would pay
    /// the drafter prefill and then pass through.
    @Test func mtpRefusesSampling() {
        #expect(Self.mtpOnly.plan(for: Self.request(temperature: 0.6)) == nil)
    }

    /// No hidden states are stored with a cached prefix, and the head needs
    /// one per prompt token.
    @Test func mtpRefusesARestoredPrefix() {
        #expect(Self.mtpOnly.plan(for: Self.request(restoresPrefix: true)) == nil)
    }

    /// The unchunked MTP prefill forfeits the boundary snapshots a thinking
    /// template's canonical leaf and a tool turn's direct-tool leaf are
    /// synthesized from; engaging would keep the conversation cold forever
    /// (the 2026-08-18 qwen3.8-27b incident, ADR-0056 amendment).
    @Test(arguments: [.canonicalUserLeaf, .directToolLeaf] as [HTTPLeafStoreMode])
    func mtpRefusesLeavesSynthesizedFromBoundaries(_ storedLeaf: HTTPLeafStoreMode) {
        #expect(Self.mtpOnly.plan(for: Self.request(storedLeaf: storedLeaf)) == nil)
    }

    /// The Raw Generation Start stores no leaf, and MTP has never engaged
    /// there (ADR-0079, considered options).
    @Test func mtpStaysOffOnTheRawGenerationStart() {
        #expect(Self.mtpOnly.plan(for: Self.request(storedLeaf: nil)) == nil)
    }

    /// 24 heads × L² × 2 bytes ≤ 4 GiB ⇒ L ≤ ~9459: the prefill is one
    /// unchunked pass, so one token past the boundary refuses.
    @Test func mtpRefusesPromptsPastTheScratchBudget() {
        let boundary = Int(
            (Double(Speculation.mtpSingleShotScratchBudgetBytes) / (24 * 2)).squareRoot())
        #expect(Self.mtpOnly.plan(for: Self.request(promptTokens: boundary))?.arm == .mtp)
        #expect(Self.mtpOnly.plan(for: Self.request(promptTokens: boundary + 1)) == nil)
    }

    @Test func mtpNeverEngagesUnpriced() {
        let unpriced = Speculation(mtpDrafter: InactiveMTPDrafter())
        #expect(unpriced.plan(for: Self.request(promptTokens: 128)) == nil)
    }

    // MARK: - Residency

    @Test func residencyNamesTheResidentDrafters() {
        #expect(!Speculation.none.isResident(.mtp))
        #expect(!Speculation.none.isResident(.dflash2))
        #expect(Self.mtpOnly.isResident(.mtp))
        #expect(!Self.mtpOnly.isResident(.dflash2))
        #expect(Self.both.isResident(.mtp) && Self.both.isResident(.dflash2))
        #expect(Self.both.memoryFacts["mtpLoaded"] == "true")
        #expect(Self.both.memoryFacts["dflash2Loaded"] == "true")
        #expect(Speculation.none.memoryFacts["dflash2WeightArrayBytes"] == "0")
    }

    /// The unload releases both drafters and reports that nothing else
    /// retains them.
    @Test func unloadReleasesTheDrafters() {
        var speculation = Speculation(
            mtpDrafter: InactiveMTPDrafter(), dflash2Drafter: Self.dflash2Drafter())
        let probe = speculation.releaseProbe
        #expect(probe.facts == ["mtpDrafterRetained": "true", "dflash2DrafterRetained": "true"])
        speculation.unload()
        #expect(!speculation.isResident(.mtp))
        #expect(!speculation.isResident(.dflash2))
        #expect(probe.facts == ["mtpDrafterRetained": "false", "dflash2DrafterRetained": "false"])
    }
}

/// What a model load attaches (ADR-0079): the load never fails for a
/// drafter, and a drafter that is off by setting, absent, refused or
/// unpaired stays off. Runs the real loader over the toy model's container.
struct SpeculationResidencyTests {

    private static func directory(_ label: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("speculation-\(label)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static let plainIdentity = ModelIdentity(
        configJSON: ["model_type": "qwen3_5"], chatTemplate: nil)

    private static func load(
        _ mode: SpeculationMode,
        checkpoint: URL,
        draftStorageRoot: URL,
        identity: ModelIdentity = plainIdentity
    ) async -> Speculation {
        let provider = ToyModelSessionProvider(model: ToyLanguageModel(script: [0]))
        return await Speculation.load(
            mode, beside: provider.container, checkpoint: checkpoint, identity: identity,
            draftStorageRoot: draftStorageRoot)
    }

    /// A draft folder that passes detection: config plus a weights file.
    private static func downloadDraft(into storageRoot: URL) throws {
        let draft = storageRoot.appendingPathComponent(DFlash2Support.draftCacheSubdirectory)
        try FileManager.default.createDirectory(at: draft, withIntermediateDirectories: true)
        try "{}".write(
            to: draft.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data([0, 1]).write(to: draft.appendingPathComponent("model.safetensors"))
    }

    @Test func offAttachesNothing() async throws {
        let root = try Self.directory("off")
        defer { try? FileManager.default.removeItem(at: root) }
        try Self.downloadDraft(into: root)
        let speculation = await Self.load(.off, checkpoint: root, draftStorageRoot: root)
        #expect(!speculation.isResident(.mtp))
        #expect(!speculation.isResident(.dflash2))
    }

    @Test func aCheckpointWithoutAnMTPHeadAttachesNoMTPDrafter() async throws {
        let checkpoint = try Self.directory("no-head")
        defer { try? FileManager.default.removeItem(at: checkpoint) }
        let speculation = await Self.load(
            .mtp, checkpoint: checkpoint, draftStorageRoot: checkpoint)
        #expect(!speculation.isResident(.mtp))
    }

    /// The head is on disk, but no MTP drafter pairs with the loaded class:
    /// pairing keys on the instance, never on the checkpoint.
    @Test func anMTPHeadNoDrafterPairsWithStaysOff() async throws {
        let checkpoint = try Self.directory("head")
        defer { try? FileManager.default.removeItem(at: checkpoint) }
        let index: [String: Any] = [
            "metadata": ["total_size": 1],
            "weight_map": ["mtp.fc.weight": "model.safetensors"],
        ]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: checkpoint.appendingPathComponent("model.safetensors.index.json"))
        #expect(MTPDrafterSupport.checkpointShipsMTPHead(directory: checkpoint))
        let speculation = await Self.load(
            .mtp, checkpoint: checkpoint, draftStorageRoot: checkpoint)
        #expect(!speculation.isResident(.mtp))
    }

    @Test func aDraftThatWasNotDownloadedStaysOff() async throws {
        let root = try Self.directory("no-draft")
        defer { try? FileManager.default.removeItem(at: root) }
        let speculation = await Self.load(.dflash2, checkpoint: root, draftStorageRoot: root)
        #expect(!speculation.isResident(.dflash2))
    }

    /// A Rotated Ternary Checkpoint refuses the draft before its geometry or
    /// weights are read (ADR-0067).
    @Test func aRotatedTernaryCheckpointRefusesTheDraft() async throws {
        let root = try Self.directory("rotated")
        defer { try? FileManager.default.removeItem(at: root) }
        try Self.downloadDraft(into: root)
        let rotated = ModelIdentity(
            configJSON: [
                "model_type": "prism_hadamard_qwen35",
                "base_model_type": "qwen3_5",
                "modules": [["path": "model.embed_tokens", "block": 1024, "embedding": true]],
            ],
            chatTemplate: nil)
        let speculation = await Self.load(
            .automatic, checkpoint: root, draftStorageRoot: root, identity: rotated)
        #expect(!speculation.isResident(.dflash2))
    }
}
