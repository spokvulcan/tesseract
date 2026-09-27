// Engine v2 contract tests — the binding contracts of ADR-0038 / spec §4,
// exercised through the public interface against scripted adapters.

import Foundation
import Testing
@testable import TesseractSpeech

private func makeEngine(
    script: ScriptedSynthesizer.Script = .init()
) async -> (SpeechEngine, ScriptedSynthesizer, RecordingLease) {
    let synth = ScriptedSynthesizer()
    await synth.configure(script)
    let lease = RecordingLease()
    let engine = SpeechEngine(
        model: .voiceDesign17B(.q8), synthesizer: synth, gpu: lease)
    return (engine, synth, lease)
}

/// ~400 words in short sentences → 3 segments at the 200-token target.
private let longText = Array(
    repeating: "The quick brown fox jumps over the lazy dog near the quiet river bank today. ",
    count: 28
).joined()

private let shortText = "Hello there, this is a short utterance."

@Suite struct EventGrammarTests {

    @Test func grammarOrderAndGaplessFrames() async throws {
        let (engine, _, _) = await makeEngine()
        let session = try await engine.session(.readAloud, voice: .standard(language: "en"))
        let utterance = try await session.speak(longText)

        #expect(utterance.segmentCount >= 2)
        #expect(utterance.sampleRate == 24_000)

        var openSegment: Int? = nil
        var lastSegmentAnnounced = -1
        var nextFrame = 0
        var finishedCount = 0
        var doneSegments: [Int] = []

        for try await event in utterance.events {
            switch event {
            case .segment(let script):
                #expect(openSegment == nil, "segment announced while another is open")
                #expect(script.index == lastSegmentAnnounced + 1, "segments in text order")
                #expect(script.startFrame == nextFrame, "startFrame is cumulative ground truth")
                #expect(!script.tokenCharOffsets.isEmpty)
                openSegment = script.index
                lastSegmentAnnounced = script.index
            case .audio(let chunk):
                #expect(chunk.segmentIndex == openSegment, "audio follows its segment event")
                #expect(chunk.frames.lowerBound == nextFrame, "frame ranges gapless")
                nextFrame = chunk.frames.upperBound
            case .segmentDone(let index):
                #expect(index == openSegment)
                openSegment = nil
                doneSegments.append(index)
            case .finished(let summary):
                finishedCount += 1
                #expect(summary.totalFrames == nextFrame)
                #expect(summary.segmentFrameCounts.count == utterance.segmentCount)
            }
        }

        #expect(finishedCount == 1, "finished exactly once on full render")
        #expect(doneSegments == Array(0..<utterance.segmentCount))
    }

    @Test func audioProjectionYieldsOnlyChunks() async throws {
        let (engine, _, _) = await makeEngine()
        let session = try await engine.session(.companion, voice: .standard(language: "en"))
        let utterance = try await session.speak(shortText)

        var chunks = 0
        for try await _ in utterance.audio { chunks += 1 }
        #expect(chunks == 3)
    }

    @Test func generationFailureTerminatesWithTypedError() async throws {
        let (engine, _, _) = await makeEngine(script: .init(failOnSegmentIndex: 0))
        let session = try await engine.session(.readAloud, voice: .standard(language: "en"))
        let utterance = try await session.speak(shortText)

        await #expect(throws: SpeechEngineError.self) {
            for try await _ in utterance.events {}
        }
    }
}

@Suite struct CancellationTests {

    @Test func consumerTaskCancelSurfacesCancellationError() async throws {
        let (engine, synth, _) = await makeEngine(
            script: .init(chunksPerSegment: 50, chunkDelayNanos: 5_000_000))
        let session = try await engine.session(.readAloud, voice: .standard(language: "en"))
        let utterance = try await session.speak(shortText)

        let consumer = Task {
            var sawCancellation = false
            do {
                var count = 0
                for try await _ in utterance.events {
                    count += 1
                    if count == 2 { withUnsafeCurrentTask { $0?.cancel() } }
                }
            } catch is CancellationError {
                sawCancellation = true
            } catch {}
            return sawCancellation
        }

        #expect(await consumer.value, "task cancel → CancellationError, untranslated")
        // Generation halted (driver cancelled, synthesizer observed it).
        try await Task.sleep(nanoseconds: 100_000_000)
        #expect(await synth.sawCancellation)
    }

    @Test func supersessionTerminatesOldStreamBeforeNewSpeakReturns() async throws {
        let (engine, _, _) = await makeEngine(
            script: .init(chunksPerSegment: 100, chunkDelayNanos: 2_000_000))
        let session = try await engine.session(.readAloud, voice: .standard(language: "en"))

        let first = try await session.speak(shortText)
        let firstResult = Task {
            do {
                for try await _ in first.events {}
                return "finished"
            } catch is CancellationError {
                return "cancelled"
            } catch {
                return "error"
            }
        }

        // Let the first utterance start producing.
        try await Task.sleep(nanoseconds: 30_000_000)
        let second = try await session.speak(shortText)
        // Contract: by the time speak() returned, the old stream terminated.
        #expect(await firstResult.value == "cancelled")

        var events = 0
        for try await _ in second.events { events += 1 }
        #expect(events > 0)
    }

    @Test func closedSessionRejectsSpeak() async throws {
        let (engine, _, _) = await makeEngine()
        let session = try await engine.session(.companion, voice: .standard(language: "en"))
        await session.close()
        await #expect(throws: SpeechEngineError.sessionClosed) {
            _ = try await session.speak(shortText)
        }
    }

    @Test func unloadTerminatesActiveUtterance() async throws {
        let (engine, _, _) = await makeEngine(
            script: .init(chunksPerSegment: 200, chunkDelayNanos: 5_000_000))
        let session = try await engine.session(.readAloud, voice: .standard(language: "en"))
        let utterance = try await session.speak(shortText)

        let consumer = Task {
            do {
                for try await _ in utterance.events {}
                return false
            } catch {
                return true
            }
        }
        try await Task.sleep(nanoseconds: 30_000_000)
        await engine.unload()
        #expect(await consumer.value, "active stream terminated by unload")
        #expect(await engine.readiness == .unloaded)
    }
}

@Suite struct PacingAndLeaseTests {

    @Test func lookaheadBoundsProductionAndLeaseIsReleasedWhileParked() async throws {
        let (engine, synth, lease) = await makeEngine()
        let session = try await engine.session(.readAloud, voice: .standard(language: "en"))
        let utterance = try await session.speak(longText)
        #expect(utterance.segmentCount >= 3)

        // Consume nothing yet: producer must stop after segment 1 (in-flight
        // finished + 1 undelivered) and park OUTSIDE the lease.
        try await Task.sleep(nanoseconds: 200_000_000)
        #expect(await synth.segmentsStarted == 1, "lookahead(1): one completed undelivered segment max")
        #expect(await lease.depth == 0, "GPU lease released while demand-parked")

        // Drain fully: production resumes segment by segment.
        var doneCount = 0
        for try await event in utterance.events {
            if case .segmentDone = event { doneCount += 1 }
        }
        #expect(doneCount == utterance.segmentCount)
        #expect(await synth.segmentsFinished == utterance.segmentCount)
        #expect(await lease.maxDepth == 1, "leases never nest")
    }

    @Test func eagerPacingRunsAhead() async throws {
        let (engine, synth, _) = await makeEngine()
        let session = try await engine.session(
            SessionProfile(reference: .none, pacing: .eager), voice: .standard(language: "en"))
        let utterance = try await session.speak(longText)

        // Without demand, eager production still completes every segment.
        try await Task.sleep(nanoseconds: 300_000_000)
        #expect(await synth.segmentsFinished == utterance.segmentCount)

        var finished = false
        for try await event in utterance.events {
            if case .finished = event { finished = true }
        }
        #expect(finished)
    }
}

@Suite struct VoiceIdentityTests {

    @Test func leadSegmentBecomesTheTakeAndEveryLaterSegmentContinuesIt() async throws {
        let (engine, synth, _) = await makeEngine()
        let session = try await engine.session(
            .readAloud, voice: .designed(description: "warm narrator", language: "en"))
        let utterance = try await session.speak(longText)
        for try await _ in utterance.events {}

        let requests = await synth.requests
        #expect(requests.count == utterance.segmentCount)
        #expect(requests[0].capturesReference)
        #expect(requests[0].reference == nil, "the take renders from the description alone")
        #expect(
            requests[0].text.split(separator: " ").count <= 40,
            "a take is short, so later segments re-read little")
        for later in requests.dropFirst() {
            #expect(later.reference?.text == requests[0].text)
            #expect(!later.capturesReference)
        }
    }

    @Test func theTakeOutlivesTheUtterance() async throws {
        let (engine, synth, _) = await makeEngine()
        let session = try await engine.session(
            .readAloud, voice: .designed(description: "warm narrator", language: "en"))
        for try await _ in try await session.speak(shortText).events {}
        for try await _ in try await session.speak(longText).events {}

        let requests = await synth.requests
        #expect(requests[0].capturesReference)
        for later in requests.dropFirst() {
            #expect(later.reference?.text == shortText, "the next utterance keeps the voice")
            #expect(!later.capturesReference)
        }
    }

    @Test func pinnedVoiceRoundTripsIntoAFreshEngine() async throws {
        let (engine, _, _) = await makeEngine()
        let session = try await engine.session(
            .companion, voice: .designed(description: "product voice", language: "en"))
        #expect(await session.exportPinnedVoice() == nil, "no take before the first segment")
        for try await _ in try await session.speak(shortText).events {}
        let exported = try #require(await session.exportPinnedVoice())
        #expect(exported.referenceText == shortText)
        #expect(exported.voiceDescription == "product voice")

        let restored = try PinnedVoice(validating: try exported.serialized())
        let (engine2, synth2, _) = await makeEngine()
        let session2 = try await engine2.session(.companion, voice: .pinned(restored))
        for try await _ in try await session2.speak(longText).events {}
        let requests = await synth2.requests
        for request in requests {
            #expect(request.reference?.codeFrames == restored.codeFrames)
            #expect(request.reference?.text == restored.referenceText)
            #expect(!request.capturesReference, "a pinned voice needs no new take")
        }
    }

    @Test func retakeReplacesTheTakeOnlyWhenItFinishes() async throws {
        let (engine, synth, _) = await makeEngine(
            script: .init(chunksPerSegment: 3, chunkDelayNanos: 20_000_000))
        let session = try await engine.session(
            .readAloud, voice: .designed(description: "warm narrator", language: "en"))
        for try await _ in try await session.speak(shortText, options: .init(seed: .fixed(1))).events {}
        let original = try #require(await session.exportPinnedVoice())

        // Cancelled mid-render: the old take stays.
        let abandoned = try await session.retake(shortText, options: .init(seed: .fixed(2)))
        let consumer = Task {
            var count = 0
            for try await _ in abandoned.events {
                count += 1
                if count == 2 { withUnsafeCurrentTask { $0?.cancel() } }
            }
        }
        _ = await consumer.result
        #expect(await session.exportPinnedVoice() == original)

        // Finished: rendered from the description alone, and it replaces the take.
        for try await _ in try await session.retake(shortText, options: .init(seed: .fixed(3))).events {}
        let last = try #require(await synth.requests.last)
        #expect(last.reference == nil)
        #expect(last.capturesReference)
        let replaced = try #require(await session.exportPinnedVoice())
        #expect(replaced != original)
        #expect(replaced.codeFrames.first?.first == 3, "the scripted take encodes its seed")

        // And the next utterance continues the new take.
        for try await _ in try await session.speak(shortText).events {}
        #expect(await synth.requests.last?.reference?.codeFrames == replaced.codeFrames)
    }

    @Test func aRetakeSpeaksOnlyTheOpening() async throws {
        let (engine, synth, _) = await makeEngine()
        let session = try await engine.session(
            .readAloud, voice: .designed(description: "warm narrator", language: "en"))
        let retake = try await session.retake(longText, options: .init(seed: .fixed(5)))
        #expect(retake.segmentCount == 1)
        for try await _ in retake.events {}
        let requests = await synth.requests
        #expect(requests.count == 1)
        #expect(requests[0].text.split(separator: " ").count <= 40)
        #expect(await session.exportPinnedVoice()?.referenceText == requests[0].text)
    }

    @Test func withoutAReferencePolicyEverySegmentStandsAlone() async throws {
        let (engine, synth, _) = await makeEngine()
        let session = try await engine.session(
            SessionProfile(reference: .none, pacing: .eager),
            voice: .designed(description: "warm narrator", language: "en"))
        for try await _ in try await session.speak(longText).events {}

        let requests = await synth.requests
        #expect(requests.allSatisfy { $0.reference == nil && !$0.capturesReference })
        #expect(await session.exportPinnedVoice() == nil)
    }

    @Test func mismatchedFingerprintIsRejected() async throws {
        let (engine, _, _) = await makeEngine()  // q8 engine
        let foreign = PinnedVoice(
            modelFingerprint: TTSModelSpec.voiceDesign17B(.q6).fingerprint,
            voiceDescription: "v", language: "en", referenceText: "v",
            codeFrames: [[1, 2, 3]])
        await #expect(throws: SpeechEngineError.self) {
            _ = try await engine.session(.companion, voice: .pinned(foreign))
        }
    }

    /// Schema 1 held a 48-frame anchor with no text; it cannot condition the
    /// in-context layout, so restoring one fails instead of re-rolling.
    @Test func schemaOneVoicesAreRejected() {
        let legacy = #"{"schema":1,"modelFingerprint":"m#q8","voiceDescription":"v","language":"en","codeFrames":[[1,2,3]]}"#
        #expect(throws: SpeechEngineError.self) {
            _ = try PinnedVoice(validating: Data(legacy.utf8))
        }
    }

    @Test func seedResolvedEntropyVariesFixedRepeats() async throws {
        let (engine, synth, _) = await makeEngine()
        let session = try await engine.session(
            SessionProfile(reference: .none, pacing: .eager), voice: .standard(language: "en"))

        for try await _ in try await session.speak(shortText, options: .init(seed: .fixed(42))).events {}
        for try await _ in try await session.speak(shortText, options: .init(seed: .fixed(42))).events {}
        for try await _ in try await session.speak(shortText, options: .init(seed: .entropy)).events {}

        let requests = await synth.requests
        #expect(requests[0].seed == 42)
        #expect(requests[1].seed == 42)
        #expect(requests[2].seed != 42 || requests[2].seed != requests[1].seed)
    }
}

/// The engine only loads a checkpoint already on disk. A missing one throws
/// `modelUnavailable` before the GPU lease, so it never blocks LLM work.
@Suite struct ModelAvailabilityTests {

    private func expectModelUnavailable(
        _ body: () async throws -> Void, sourceLocation: SourceLocation = #_sourceLocation
    ) async {
        let error = await #expect(throws: SpeechEngineError.self, sourceLocation: sourceLocation) {
            try await body()
        }
        guard let error else { return }  // #expect recorded the miss
        guard case .modelUnavailable = error else {
            Issue.record("expected modelUnavailable, got \(String(describing: error))",
                sourceLocation: sourceLocation)
            return
        }
    }

    @Test func missingCheckpointFailsBeforeTheLease() async throws {
        let (engine, synth, lease) = await makeEngine()
        await synth.setCheckpointOnDisk(false)

        await expectModelUnavailable {
            _ = try await engine.session(.readAloud, voice: .standard(language: "en"))
        }
        await expectModelUnavailable { try await engine.prepare(.warm) }

        #expect(await lease.acquisitions == 0, "no GPU lease for a model that isn't there")
        #expect(await synth.loadCount == 0)
        #expect(await engine.readiness == .unloaded)
    }

    @Test func checkpointArrivingLaterLoadsNormally() async throws {
        let (engine, synth, _) = await makeEngine()
        await synth.setCheckpointOnDisk(false)
        await expectModelUnavailable {
            _ = try await engine.session(.companion, voice: .standard(language: "en"))
        }

        // The download finishes; the failure left nothing behind to clear.
        await synth.setCheckpointOnDisk(true)
        let session = try await engine.session(.companion, voice: .standard(language: "en"))
        for try await _ in try await session.speak(shortText).events {}

        #expect(await synth.loadCount == 1)
        #expect(await engine.readiness == .warm)
    }

    @Test func concurrentOpensShareOneCheck() async throws {
        let (engine, synth, _) = await makeEngine()
        async let a: Void = engine.prepare(.loaded)
        async let b: Void = engine.prepare(.loaded)
        _ = try await (a, b)
        #expect(await synth.availabilityChecks == 1, "the check rides the coalesced load")
    }
}

@Suite struct LifecycleTests {

    @Test func lazyLoadOnFirstUseThenWarm() async throws {
        let (engine, synth, _) = await makeEngine()
        #expect(await engine.readiness == .unloaded)
        let session = try await engine.session(.readAloud, voice: .standard(language: "en"))
        #expect(await engine.readiness == .warm, "load + warmup happen at session open")
        #expect(await synth.loadCount == 1)
        #expect(await synth.warmUpCount == 1)
        _ = session
    }

    @Test func prepareCoalescesAndPrimes() async throws {
        let (engine, synth, _) = await makeEngine()
        async let a: Void = engine.prepare(.warm, priming: [.designed(description: "narrator", language: "en")])
        async let b: Void = engine.prepare(.warm)
        _ = try await (a, b)
        #expect(await synth.loadCount == 1, "concurrent prepares coalesce")
        #expect(await synth.primedVoices.contains("narrator"))
    }

    @Test func utteranceEndTrimsCaches() async throws {
        let (engine, synth, _) = await makeEngine()
        let session = try await engine.session(.companion, voice: .standard(language: "en"))
        for try await _ in try await session.speak(shortText).events {}
        try await Task.sleep(nanoseconds: 50_000_000)
        #expect(await synth.trimCount >= 1, "one cache trim per utterance end (ADR-0039)")
    }
}

@Suite struct SegmenterTests {

    @Test func aCapturingUtteranceLeadsWithAShortSegment() {
        let plain = Segmenter.segment(longText)
        let led = Segmenter.segment(longText, leadTokens: Segmenter.referenceLeadTokens)
        #expect(led[0].text.split(separator: " ").count <= 40)
        #expect(led[0].text.count < plain[0].text.count)
        #expect(
            led.map(\.text).joined() == plain.map(\.text).joined(),
            "no text is lost or reordered")
    }

    @Test func aTitleNeverBecomesTheWholeTake() {
        let text =
            "Chapter One. "
            + String(
                repeating: "The rain had stopped by the time she reached the quiet harbor. ",
                count: 6)
        let led = Segmenter.segment(text, leadTokens: Segmenter.referenceLeadTokens)
        #expect(led[0].text.hasPrefix("Chapter One."))
        #expect(led[0].text.split(separator: " ").count > 10)
    }

    @Test func withoutALeadTheSegmentsAreUnchanged() {
        #expect(Segmenter.segment(longText, leadTokens: nil) == Segmenter.segment(longText))
    }
}

@Suite struct SilenceCapTests {
    /// Qwen3-TTS's geometry: 1,920 samples per 80 ms frame, so the 1.2 s cap
    /// is 15 frames.
    private static let format = AudioFormat(sampleRate: 24_000, samplesPerFrame: 1920)
    private static let frame = format.samplesPerFrame

    private static func frames(_ count: Int, level: Float) -> [Float] {
        [Float](repeating: level, count: count * frame)
    }

    private static let speech: Float = 0.05
    private static let silence: Float = 0.0001

    @Test func speechAndReadersPausesPassUnchanged() {
        var cap = SilenceCap(format: Self.format)
        let input =
            Self.frames(3, level: Self.speech) + Self.frames(10, level: Self.silence)
            + Self.frames(2, level: Self.speech)
        #expect(cap.apply(input) == input)
    }

    @Test func aStallIsCutToTheCapAndNoSpeechIsLost() {
        var cap = SilenceCap(format: Self.format)
        let input =
            Self.frames(2, level: Self.speech) + Self.frames(150, level: Self.silence)
            + Self.frames(4, level: Self.speech)
        let kept = cap.apply(input)
        #expect(kept.count == (2 + 15 + 4) * Self.frame)
        #expect(kept.filter { $0 == Self.speech }.count == 6 * Self.frame)
    }

    @Test func theRunCarriesAcrossChunks() {
        var cap = SilenceCap(format: Self.format)
        let first = cap.apply(Self.frames(10, level: Self.silence))
        let second = cap.apply(Self.frames(10, level: Self.silence))
        let resumed = cap.apply(Self.frames(1, level: Self.speech) + Self.frames(3, level: Self.silence))
        #expect(first.count + second.count == 15 * Self.frame)
        #expect(resumed.count == 4 * Self.frame, "sound resets the run")
    }
}
