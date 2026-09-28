//
//  DictationLabVoiceFix.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  Variant E: hold the fix shortcut and say the correction. Its own capture
//  session on the shared microphone (the same shape Voice Input uses), so a
//  spoken fix never becomes a dictation. What was said is matched against
//  the last take:
//
//  - "X to Y" (or "change X to Y"): X becomes Y;
//  - letters ("C L A U D E"): the word they spell replaces the word that
//    sounds most like it;
//  - a short phrase: replaces the span that sounds most like it;
//  - most of the sentence again: replaces the whole take.
//
//  Aqua Voice's Edit Mode is the precedent (select, hold, say the fix).
//

import AppKit
import Observation

@Observable @MainActor
final class LabVoiceFix {
    enum State: Equatable {
        case idle
        case listening
        case resolving
        case failed(String)
    }

    private(set) var state: State = .idle
    /// The last thing said, and what it did.
    private(set) var lastHeard: String?

    @ObservationIgnored weak var lab: DictationLab?
    @ObservationIgnored private let session: VoiceCaptureSession
    @ObservationIgnored private let settings: SettingsManager

    init(
        audioCapture: any AudioCapturing, transcriptionEngine: any Transcribing,
        settings: SettingsManager
    ) {
        self.session = VoiceCaptureSession(
            audioCapture: audioCapture, transcriptionEngine: transcriptionEngine)
        self.settings = settings
    }

    func begin() {
        guard state != .listening, state != .resolving else { return }
        guard lab?.lastTake != nil else {
            lab?.showToast(LabToast(title: "Nothing to fix yet", detail: "Dictate something first"))
            return
        }
        switch session.start() {
        case .started:
            state = .listening
            if let lab { SayItAgainFlow.showListening(lab) }
        case .micBusy: fail("The microphone is busy")
        case .captureFailed: fail("Couldn't start the microphone")
        }
    }

    func end() {
        guard state == .listening else { return }
        switch session.stop() {
        case .noAudio, .tooShort:
            state = .idle
            lab?.panels.closeCard()
            return
        case .audio(let audio, _):
            state = .resolving
            Task {
                var heard = ""
                _ = await session.transcribeAndCommit(audio, language: settings.language) {
                    text, _ in
                    heard = text
                }
                await resolve(heard)
            }
        }
    }

    private func fail(_ message: String) {
        state = .failed(message)
        if let lab { SayItAgainFlow.showListening(lab) }
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(2.5))
            if case .failed = self?.state {
                self?.state = .idle
                self?.lab?.panels.closeCard()
            }
        }
    }

    private func resolve(_ heard: String) async {
        let said = heard.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: ".!?"))
        lastHeard = said
        guard let lab, let take = lab.lastTake, !said.isEmpty else {
            state = .idle
            return
        }
        guard let corrected = Self.apply(said, to: take.text) else {
            fail("Couldn't tell which words \"\(said)\" fixes")
            return
        }
        state = .idle
        // The shortcut press and its release are the fix's own keys.
        await lab.fix(take.id, to: corrected, source: .voice, allowedKeys: 0)
    }

    /// The take with the spoken fix applied, or nil when nothing matches.
    static func apply(_ said: String, to text: String) -> String? {
        var words = text.split(separator: " ").map(String.init)
        guard !words.isEmpty else { return nil }

        // "X to Y", "change X to Y", "replace X with Y"
        if let (from, to) = command(said) {
            if let range = bestSpan(for: from, in: words, threshold: 0.55, allowExact: true) {
                return replacing(range, in: &words, with: to)
            }
            return nil
        }
        // Spelled letters: "C L A U D E", "C-L-A-U-D-E"
        let letters = said.split(whereSeparator: { " ,-.".contains($0) })
        if letters.count >= 3, letters.allSatisfy({ $0.count == 1 && $0.first!.isLetter }) {
            let spelled = letters.joined()
            let word = spelled.prefix(1).uppercased() + spelled.dropFirst().lowercased()
            if let range = bestSpan(for: word, in: words, threshold: 0.4, maxLength: 2) {
                return replacing(range, in: &words, with: word)
            }
            return nil
        }
        // Most of the sentence again: a re-dictation.
        let saidWords = said.split(separator: " ")
        if saidWords.count >= max(4, Int(Double(words.count) * 0.6)) {
            return said
        }
        // A short phrase: the span that sounds most like it.
        if let range = bestSpan(for: said, in: words, threshold: 0.5) {
            return replacing(range, in: &words, with: said)
        }
        return nil
    }

    private static func command(_ said: String) -> (String, String)? {
        let lower = said.lowercased()
        for prefix in ["change ", "correct ", "replace ", "fix "] where lower.hasPrefix(prefix) {
            return command(String(said.dropFirst(prefix.count)))
        }
        for separator in [" with ", " to ", " into "] {
            if let r = said.range(of: separator, options: .caseInsensitive) {
                let from = said[..<r.lowerBound].trimmingCharacters(in: .whitespaces)
                let to = said[r.upperBound...].trimmingCharacters(in: .whitespaces)
                if !from.isEmpty, !to.isEmpty, from.split(separator: " ").count <= 3 {
                    return (from, to)
                }
            }
        }
        return nil
    }

    private static func bestSpan(
        for phrase: String, in words: [String], threshold: Double, maxLength: Int? = nil,
        allowExact: Bool = false
    ) -> Range<Int>? {
        let n = phrase.split(separator: " ").count
        var best: (Range<Int>, Double)?
        let lengths = Set([max(1, n - 1), n, n + 1].filter { $0 <= (maxLength ?? .max) })
        for length in lengths where length <= words.count {
            for start in 0...(words.count - length) {
                let span = words[start..<(start + length)].joined(separator: " ")
                let bare = span.trimmingCharacters(in: .punctuationCharacters)
                if bare.lowercased() == phrase.lowercased() {
                    if allowExact { return start..<(start + length) }
                    continue
                }
                let score = Phonetic.similarity(bare, phrase)
                if score > (best?.1 ?? threshold) { best = (start..<(start + length), score) }
            }
        }
        return best?.0
    }

    private static func replacing(
        _ range: Range<Int>, in words: inout [String], with phrase: String
    ) -> String {
        // Keep the punctuation that ended the replaced span.
        let last = words[range.upperBound - 1]
        let trailing = String(last.reversed().prefix(while: { $0.isPunctuation }).reversed())
        words.replaceSubrange(range, with: [phrase + trailing])
        return words.joined(separator: " ")
    }
}
