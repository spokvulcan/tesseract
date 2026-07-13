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
            SessionProfile(anchor: .none, pacing: .eager), voice: .standard(language: "en"))
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

    @Test func perUtteranceAnchorCapturedOnSegmentZeroAndUsedAfter() async throws {
        let (engine, synth, _) = await makeEngine()
        let session = try await engine.session(
            .readAloud, voice: .designed(description: "warm narrator", language: "en"))
        let utterance = try await session.speak(longText)
        for try await _ in utterance.events {}

        let requests = await synth.requests
        #expect(requests.count == utterance.segmentCount)
        #expect(requests[0].captureAnchorSteps == 48)
        #expect(requests[0].anchor == nil)
        for later in requests.dropFirst() {
            #expect(later.anchor != nil, "later segments conditioned on the captured anchor")
            #expect(later.captureAnchorSteps == nil)
        }

        // Per-utterance anchor dies with the utterance: a second utterance re-captures.
        let second = try await session.speak(shortText)
        for try await _ in second.events {}
        let secondFirst = await synth.requests[utterance.segmentCount]
        #expect(secondFirst.captureAnchorSteps == 48)
        #expect(secondFirst.anchor == nil)
    }

    @Test func pinnedAnchorPersistsAcrossUtterancesAndExports() async throws {
        let (engine, synth, _) = await makeEngine()
        let session = try await engine.session(
            .companion, voice: .designed(description: "product voice", language: "en"))

        let first = try await session.speak(shortText)
        for try await _ in first.events {}
        let exported = await session.exportPinnedVoice()
        #expect(exported != nil, "anchor formed on first utterance is exportable")

        let second = try await session.speak(shortText)
        for try await _ in second.events {}
        let requests = await synth.requests
        #expect(requests[1].anchor != nil, "pinned session reuses the anchor")
        #expect(requests[1].captureAnchorSteps == nil)

        // Round-trip: restore into a fresh engine.
        let data = try exported!.serialized()
        let restored = try PinnedVoice(validating: data)
        let (engine2, synth2, _) = await makeEngine()
        let session2 = try await engine2.session(.companion, voice: .pinned(restored))
        let third = try await session2.speak(shortText)
        for try await _ in third.events {}
        let firstRequest = await synth2.requests[0]
        #expect(firstRequest.anchor?.codeFrames == restored.codeFrames)
        #expect(firstRequest.captureAnchorSteps == nil, "restored voice needs no capture")
    }

    @Test func mismatchedFingerprintIsRejected() async throws {
        let (engine, _, _) = await makeEngine()  // q8 engine
        let foreign = PinnedVoice(
            modelFingerprint: TTSModelSpec.voiceDesign17B(.q6).fingerprint,
            voiceDescription: "v", language: "en",
            codeFrames: [[1, 2, 3]])
        await #expect(throws: SpeechEngineError.self) {
            _ = try await engine.session(.companion, voice: .pinned(foreign))
        }
    }

    @Test func seedResolvedEntropyVariesFixedRepeats() async throws {
        let (engine, synth, _) = await makeEngine()
        let session = try await engine.session(
            SessionProfile(anchor: .none, pacing: .eager), voice: .standard(language: "en"))

        for try await _ in try await session.speak(shortText, options: .init(seed: .fixed(42))).events {}
        for try await _ in try await session.speak(shortText, options: .init(seed: .fixed(42))).events {}
        for try await _ in try await session.speak(shortText, options: .init(seed: .entropy)).events {}

        let requests = await synth.requests
        #expect(requests[0].seed == 42)
        #expect(requests[1].seed == 42)
        #expect(requests[2].seed != 42 || requests[2].seed != requests[1].seed)
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
