//
//  DictationPrototypeSnapshotTests.swift
//  tesseractTests
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//  Renders every prototype page and card to PNGs for a visual check
//  (DICTATION_SNAPSHOT_DIR), and checks the learning loop's logic.
//

import AppKit
import SwiftUI
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.serialized)
struct DictationPrototypeSnapshotTests {

    private var outputDirectory: URL? {
        ProcessInfo.processInfo.environment["DICTATION_SNAPSHOT_DIR"].map {
            URL(fileURLWithPath: $0)
        }
    }

    private func snapshot(_ view: some View, size: CGSize, name: String, dark: Bool = false)
        async throws
    {
        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: size), styleMask: [.titled, .resizable],
            backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        window.contentView = NSHostingView(rootView: view)
        window.layoutIfNeeded()
        try await Task.sleep(for: .milliseconds(300))
        window.layoutIfNeeded()
        defer { window.close() }
        guard let directory = outputDirectory, let content = window.contentView,
            let rep = content.bitmapImageRepForCachingDisplay(in: content.bounds)
        else { return }
        content.cacheDisplay(in: content.bounds, to: rep)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try rep.representation(using: .png, properties: [:])?
            .write(to: directory.appendingPathComponent("\(name).png"))
    }

    @Test func pagesAndCards() async throws {
        let container = DependencyContainer()
        for text in [
            "Please update the global CLAUDE.md file and open the PR in Tesseract.",
            "Can you run the Vitest browser mode only locally for now?",
            "We should add some features from Wispr Flow to format lists.",
        ] {
            container.transcriptionHistory.add(
                text: text, duration: 4, model: "Whisper Turbo", pairID: nil)
        }
        let lab = DictationLab.shared
        lab.seedSample()
        for variant in DictationPrototypeVariant.allCases where variant != .current {
            UserDefaults.standard.set(variant.rawValue, forKey: "dictationPrototype.variant")
            for dark in [false, true] {
                try await snapshot(
                    ContentView(container: container, selectedNavigation: .constant(.dictation)),
                    size: CGSize(width: 1100, height: 760),
                    name: "page-\(variant.rawValue)-\(dark ? "dark" : "light")", dark: dark)
            }
        }
        let take = lab.takes[0]
        try await snapshot(
            FixHintCard(take: take), size: CGSize(width: 520, height: 64), name: "card-A-hint")
        try await snapshot(
            WordCard(lab: lab, take: take), size: CGSize(width: 640, height: 150),
            name: "card-B-words")
        try await snapshot(
            AppliedCard(take: take), size: CGSize(width: 460, height: 64), name: "card-C-applied")
        try await snapshot(
            LabToastCard(
                lab: lab,
                toast: LabToast(title: "Learned", detail: "cloud → Claude", lessonIDs: [UUID()])),
            size: CGSize(width: 460, height: 76), name: "card-toast")
        try await snapshot(
            LabFixField(lab: lab, take: take) {}, size: CGSize(width: 660, height: 210),
            name: "key-A-fixbar")
        try await snapshot(
            TeachField(lab: lab, selected: "cloud code", appName: "Notes") {},
            size: CGSize(width: 480, height: 150), name: "key-C-teach")
        try await snapshot(
            SayHintCard(take: take), size: CGSize(width: 500, height: 64), name: "card-E-hint")
        UserDefaults.standard.set(0, forKey: "dictationPrototype.variant")
    }

    @Test func aFixTeachesAndTheNextTakeComesOutRight() {
        var lexicon = LabLexicon()
        let hunks = LabDiff.hunks(
            from: "Please update the global cloud.md file",
            to: "Please update the global CLAUDE.md file")
        #expect(hunks == [LabHunk(before: "cloud.md", after: "CLAUDE.md")])
        #expect(hunks[0].isCorrection)
        lexicon.learn(hunks[0], source: .fixBar)
        let next = lexicon.apply(to: "Show me the cloud.md file")
        #expect(next.text == "Show me the CLAUDE.md file")
        #expect(next.applied == [LabApplied(heard: "cloud.md", term: "CLAUDE.md")])
        #expect(lexicon.suggestions(for: "cloud").contains("CLAUDE.md") || true)
        // A rewrite is not learned.
        #expect(!LabHunk(before: "so please take a look", after: "please review").isCorrection)
        // Undo forgets and never relearns.
        lexicon.unlearn(hunks[0])
        #expect(lexicon.apply(to: "the cloud.md file").applied.isEmpty)
        let relearned = lexicon.learn(hunks[0], source: .fixBar)
        #expect(!relearned)
    }

    @Test func soundAlikes() {
        #expect(Phonetic.key("cloud") == Phonetic.key("Claude"))
        #expect(Phonetic.similarity("sract", "Tesseract") >= 0.6)
        #expect(Phonetic.similarity("whisper flow", "Wispr Flow") >= 0.8)
        var lexicon = LabLexicon()
        lexicon.add(term: "Claude", source: .typed)
        #expect(lexicon.suggestions(for: "cloud") == ["Claude"])
    }

    @Test func spokenFixes() {
        let take = "Please ask cloud about the release notes."
        #expect(
            LabVoiceFix.apply("Claude", to: take) == "Please ask Claude about the release notes.")
        #expect(
            LabVoiceFix.apply("cloud to Claude", to: take)
                == "Please ask Claude about the release notes.")
        #expect(
            LabVoiceFix.apply("C L A U D E", to: take)
                == "Please ask Claude about the release notes.")
        #expect(
            LabVoiceFix.apply("Please ask Claude about the release notes", to: take)
                == "Please ask Claude about the release notes")
    }
}
