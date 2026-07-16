//
//  CompanionVoiceDemo.swift
//  tesseract
//

import Foundation
import SwiftUI

// MARK: - Scenes

/// PROTOTYPE (map #301, ticket #328) — one scripted anchor-day moment the
/// overlay concepts perform. The scripts are theatre: their job is to push a
/// concept through every face (summons escalation, turn-taking, thinking,
/// the contract stamp, dissolve) with anchor-true content, so the owner
/// reacts to choreography, not to lorem ipsum.
struct CompanionVoiceScene: Identifiable, Sendable {
    enum Step: Sendable {
        /// Knock and wait: escalates every few seconds; a click engages,
        /// `autoEngageAfter` keeps an unattended demo moving.
        case summon(line: String, autoEngageAfter: TimeInterval)
        /// The Companion speaks (words reveal at talk pace; click = barge-in).
        case speak(String)
        /// The owner talks; the mock partial builds in ASR-ish bursts, then
        /// dwells before the turn settles.
        case listen(owner: String, dwell: TimeInterval)
        case think(TimeInterval)
        /// Stamp the day contract chip (anchor #302's one hard step).
        case contract(String)
        case pause(TimeInterval)
    }

    let id: String
    let title: String
    let steps: [Step]

    /// The wake-linked summons + ~2-min planning talk, condensed.
    static let morning = CompanionVoiceScene(
        id: "morning", title: "Morning",
        steps: [
            .summon(line: "Morning. Time to plan the day.", autoEngageAfter: 14),
            .speak(
                "Morning. Six hours — rough night. Yesterday you promised the retention deck would be today's fight."
            ),
            .listen(
                owner: "Yeah. There's the deck, two interviews, and I owe Maya that review.",
                dwell: 0.9),
            .think(1.4),
            .speak(
                "The deck is the hard one. The interviews are fixed points, and Maya can wait until four."
            ),
            .listen(owner: "Deal. Deck until noon, nothing else before it.", dwell: 0.8),
            .contract("Ship the retention deck draft by noon."),
            .speak("Held. I'll knock at half past one."),
            .pause(1.6),
        ])

    /// The ~20-second midday pulse: quiet-first, names drift once, sides with
    /// momentum.
    static let midday = CompanionVoiceScene(
        id: "midday", title: "Midday pulse",
        steps: [
            .summon(line: "Half past one — quick pulse.", autoEngageAfter: 8),
            .speak("The deck. Still moving?"),
            .listen(owner: "Slipped — support pulled me in for an hour.", dwell: 0.8),
            .speak(
                "Then that was the drift, and it's named. The deck is still the day — back to it, I'll hold the door."
            ),
            .pause(1.4),
        ])

    /// The evening freeform journal with gap-fill follow-ups and the contract
    /// close — the contract is always closed, never abandoned (anchor #302).
    static let evening = CompanionVoiceScene(
        id: "evening", title: "Evening",
        steps: [
            .summon(line: "Evening. Put the day away?", autoEngageAfter: 10),
            .speak("Evening. Talk me through it — how did the day actually go?"),
            .listen(
                owner: "Deck went out around two, later than we said but it went. "
                    + "Interviews were fine — the second one was genuinely strong. "
                    + "Never touched Maya's review.",
                dwell: 1.2),
            .think(1.6),
            .speak(
                "Then the contract closed — late, but shipped beats noon. Maya's review rolls to tomorrow morning, first thing."
            ),
            .contract("Closed: retention deck shipped 2:10 pm."),
            .speak(
                "One more thing before I put it away — anything worth keeping from those interviews?"
            ),
            .listen(owner: "Just that the second candidate is the bar now.", dwell: 0.9),
            .speak("Kept. Good night, sir."),
            .pause(1.8),
        ])

    static let all: [CompanionVoiceScene] = [.morning, .midday, .evening]
}

// MARK: - Driver

/// PROTOTYPE (map #301, ticket #328) — walks a scene's steps and performs
/// them into the `CompanionVoiceFeed`: word-paced speech reveal, bursty mock
/// ASR partials, a ~30 Hz synthetic meter, summons escalation with
/// auto-engage, click barge-in with a fixed interjection exchange. All
/// theatre, no engines.
@MainActor
final class CompanionVoiceDemoDriver {

    private let feed: CompanionVoiceFeed
    /// Scene lifecycle hook — the prototype controller shows/tears down the
    /// panel around it.
    var onActiveChange: ((Bool) -> Void)?

    private var runTask: Task<Void, Never>?
    private var meterTask: Task<Void, Never>?
    private var engaged = false
    private var bargedIn = false
    private var dismissed = false

    init(feed: CompanionVoiceFeed) {
        self.feed = feed
    }

    var isRunning: Bool { runTask != nil }

    func play(_ scene: CompanionVoiceScene) {
        stop()
        feed.reset()
        feed.setScene(title: scene.title)
        onActiveChange?(true)
        Log.companion.info("Voice-overlay demo: playing scene '\(scene.id)'")
        runTask = Task { [weak self] in
            await self?.run(scene)
            self?.finish()
        }
    }

    /// The concept views' click surface (via `CompanionVoiceActions`).
    func engage() { engaged = true }

    func bargeIn() {
        guard case .speaking = feed.state else { return }
        bargedIn = true
    }

    func dismiss() {
        Log.companion.info("Voice-overlay demo: dismissed")
        dismissed = true
        engaged = true
    }

    func stop() {
        runTask?.cancel()
        runTask = nil
        stopMeter()
        feed.reset()
    }

    private func finish() {
        runTask = nil
        stopMeter()
        withMainAnimation { feed.setState(.idle) }
        onActiveChange?(false)
    }

    // MARK: - Performance

    private func run(_ scene: CompanionVoiceScene) async {
        engaged = false
        dismissed = false
        for step in scene.steps {
            if Task.isCancelled || dismissed { return }
            switch step {
            case .summon(let line, let autoEngageAfter):
                await performSummons(line: line, autoEngageAfter: autoEngageAfter)
            case .speak(let text):
                await performSpeech(text)
                if bargedIn { await performBargeIn() }
            case .listen(let owner, let dwell):
                await performListen(owner: owner, dwell: dwell)
            case .think(let seconds):
                withMainAnimation { feed.setState(.thinking) }
                await sleep(seconds)
            case .contract(let text):
                withMainAnimation { feed.stampContract(text) }
                await sleep(1.2)
            case .pause(let seconds):
                await sleep(seconds)
            }
        }
    }

    private func performSummons(line: String, autoEngageAfter: TimeInterval) async {
        engaged = false
        feed.setSummons(line)
        withMainAnimation { feed.setState(.summoning(escalation: 0)) }
        let start = Date()
        var escalation = 0
        while !engaged && !Task.isCancelled {
            let elapsed = Date().timeIntervalSince(start)
            if elapsed >= autoEngageAfter { break }
            let due = min(2, Int(elapsed / 4))
            if due != escalation {
                escalation = due
                withMainAnimation { feed.setState(.summoning(escalation: due)) }
            }
            await sleep(0.1)
        }
    }

    private func performSpeech(_ text: String) async {
        bargedIn = false
        feed.beginSpokenLine(text)
        withMainAnimation { feed.setState(.speaking) }
        startMeter(voice: .companion)
        for word in feed.spokenWords {
            if Task.isCancelled || bargedIn || dismissed { break }
            withMainAnimation(.easeOut(duration: 0.18)) { feed.revealNextWord() }
            await sleep(wordDelay(after: word))
        }
        stopMeter()
        guard !Task.isCancelled else { return }
        // The line settles into the transcript whole — even a barged-in line
        // was said up to the interruption.
        feed.settle(role: .companion, text: feed.revealedText)
        if !bargedIn { await sleep(0.5) }
    }

    /// A fixed interjection exchange, so a barge-in click shows the real
    /// choreography: Companion yields instantly, owner talks, Companion
    /// acknowledges, then the scene resumes.
    private func performBargeIn() async {
        Log.companion.info("Voice-overlay demo: barge-in")
        await performListen(
            owner: "Hold on — move that after lunch, the morning is gone already.", dwell: 0.7)
        await performSpeech("Understood — after lunch it is.")
        bargedIn = false
    }

    private func performListen(owner: String, dwell: TimeInterval) async {
        withMainAnimation { feed.setState(.listening) }
        startMeter(voice: .owner)
        let words = owner.split(separator: " ").map(String.init)
        var shown = 0
        var generator = SplitMix64(seed: UInt64(words.count) &* 0x9E37_79B9)
        while shown < words.count && !Task.isCancelled && !dismissed {
            // ASR-ish burst: a few words at a time, at an uneven cadence.
            shown = min(words.count, shown + Int(generator.next(in: 2...4)))
            feed.setPartial(words.prefix(shown).joined(separator: " "))
            await sleep(Double(generator.next(in: 38...75)) / 100)
        }
        await sleep(dwell)
        stopMeter()
        guard !Task.isCancelled else { return }
        feed.settle(role: .owner, text: owner)
        feed.setPartial(nil)
    }

    // MARK: - Synthetic meter

    private enum Voice { case companion, owner }

    /// ~30 Hz layered-sine loudness + bands: enough life for waveforms and
    /// breath choreography without touching a microphone.
    private func startMeter(voice: Voice) {
        stopMeter()
        let start = Date()
        meterTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let t = Date().timeIntervalSince(start)
                let base: Double = voice == .companion ? 0.52 : 0.44
                let level =
                    base
                    + 0.22 * sin(t * 3.1) * sin(t * 0.9)
                    + 0.14 * sin(t * 7.7 + 1.3)
                    + 0.08 * sin(t * 17.3)
                var bands = [Float](repeating: 0, count: MeterFrame.bandCount)
                for band in 0..<bands.count {
                    let phase = Double(band) * 0.8
                    let falloff = 1.0 - Double(band) / Double(bands.count) * 0.55
                    let value = max(
                        0,
                        level * falloff + 0.18 * sin(t * (5.0 + Double(band) * 1.7) + phase))
                    bands[band] = Float(min(1, value))
                }
                self.feed.setMeter(level: Float(max(0.05, min(1, level))), spectrum: bands)
                try? await Task.sleep(for: .milliseconds(33))
            }
        }
    }

    private func stopMeter() {
        meterTask?.cancel()
        meterTask = nil
        feed.setMeter(level: 0, spectrum: MeterFrame.zeroBands)
    }

    // MARK: - Pacing

    /// Talk pace: ~180 wpm with punctuation breathing room.
    private func wordDelay(after word: String) -> TimeInterval {
        if word.hasSuffix(".") || word.hasSuffix("?") || word.hasSuffix("!") { return 0.62 }
        if word.hasSuffix(",") || word.hasSuffix(";") || word.hasSuffix("—") { return 0.44 }
        return 0.30
    }

    private func sleep(_ seconds: TimeInterval) async {
        try? await Task.sleep(for: .milliseconds(Int(seconds * 1000)))
    }
}

// MARK: - Small helpers

/// `withAnimation` under a name that reads as "state flip the views spring on".
@MainActor
private func withMainAnimation(
    _ animation: SwiftUI.Animation = .spring(duration: 0.45), _ body: () -> Void
) {
    SwiftUI.withAnimation(animation, body)
}

/// Tiny deterministic generator — the demo must feel organic, not be random.
private struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func next(in range: ClosedRange<Int>) -> Int {
        range.lowerBound + Int(next() % UInt64(range.count))
    }
}
