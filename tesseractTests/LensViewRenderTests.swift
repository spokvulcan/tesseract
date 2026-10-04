//
//  LensViewRenderTests.swift
//  tesseractTests
//
//  Renders the **Lens** card (PRD #612) in each of its states the way the
//  panel hosts it, and checks that it fits the card: opened on a take,
//  typing with a target and its completion, a word picked by hand, nothing
//  that sounds like it, done with what it learned, and done when the app
//  moved on. The repo has no image snapshot suite, so these tests check
//  that each state lays out within the Lens's size limits. They don't
//  compare pixels. Set `TEST_RUNNER_LENS_RENDER_DIR` to also write a PNG of
//  each state for a person to look at.
//

import AppKit
import SwiftUI
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.serialized)
struct LensViewRenderTests {

    private static let terminal = TargetApp(
        bundleID: "com.apple.Terminal", name: "Terminal", pid: 7)

    private func makeModel() -> LensModel {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("lens-render-\(UUID().uuidString)", isDirectory: true)
        return LensModel(
            learnedWords: LearnedWordStore(directory: directory), pairs: nil, history: nil)
    }

    private func take(_ text: String) -> DictatedTake {
        DictatedTake(pairID: nil, text: text, catches: [], app: Self.terminal, pasted: true)
    }

    /// Lays the card out at the panel's width and returns its height,
    /// writing a PNG when a render directory is set.
    private func render(_ model: LensModel, named name: String) async throws -> CGFloat {
        var height: CGFloat = 0
        let view = LensView(model: model, actions: .none, onHeightChange: { height = $0 })
            .background(Color(nsColor: .windowBackgroundColor))
        let hosting = NSHostingView(rootView: view)
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: LensStyle.width, height: LensStyle.maxHeight),
            styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = hosting
        defer { window.close() }
        window.layoutIfNeeded()
        try await Task.sleep(for: .milliseconds(50))
        window.layoutIfNeeded()

        let fitting = hosting.fittingSize
        if let directory = ProcessInfo.processInfo.environment["LENS_RENDER_DIR"] {
            hosting.frame = NSRect(origin: .zero, size: fitting)
            hosting.layoutSubtreeIfNeeded()
            if let bitmap = hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds) {
                hosting.cacheDisplay(in: hosting.bounds, to: bitmap)
                let url = URL(fileURLWithPath: directory).appendingPathComponent("lens-\(name).png")
                try bitmap.representation(using: .png, properties: [:])?.write(to: url)
            }
        }
        return max(height, fitting.height)
    }

    private func expectFits(_ height: CGFloat, sourceLocation: SourceLocation = #_sourceLocation) {
        #expect(height > 40, sourceLocation: sourceLocation)
        #expect(height <= LensStyle.maxHeight, sourceLocation: sourceLocation)
    }

    @Test func openedOnATake() async throws {
        let model = makeModel()
        model.open(
            take("Ask cloud why the SRACT server drops the first request."), mode: .afterPaste,
            vocabulary: ["Claude", "Tesseract"])
        expectFits(try await render(model, named: "opened"))
    }

    @Test func typingWithATargetAndACompletion() async throws {
        let model = makeModel()
        model.open(
            take("Ask cloud why the SRACT server drops the first request."), mode: .afterPaste,
            vocabulary: ["Claude", "Tesseract"])
        model.typed = "tess"
        #expect(model.target != nil)
        expectFits(try await render(model, named: "typing"))
    }

    @Test func aWordPickedByHand() async throws {
        let model = makeModel()
        model.open(take("Ship the build tonight."), mode: .afterPaste, vocabulary: [])
        model.moveTarget(by: -1)
        expectFits(try await render(model, named: "picked"))
    }

    @Test func nothingSoundsLikeIt() async throws {
        let model = makeModel()
        model.open(take("Open the door."), mode: .afterPaste, vocabulary: [])
        model.typed = "Tesseract"
        #expect(model.target == nil)
        expectFits(try await render(model, named: "no-target"))
    }

    @Test func doneWithWhatItLearned() async throws {
        let model = makeModel()
        model.open(take("Ask cloud why."), mode: .afterPaste, vocabulary: ["Claude"])
        model.typed = "claude"
        #expect(model.commit())
        model.finish(LensModel.Result(line: "Fixed in Terminal", detail: nil))
        expectFits(try await render(model, named: "done"))
    }

    @Test func doneWhenTheAppMovedOn() async throws {
        let model = makeModel()
        model.open(take("Ask cloud why."), mode: .afterPaste, vocabulary: ["Claude"])
        model.typed = "claude"
        #expect(model.commit())
        model.finish(
            LensModel.Result(
                line: "Terminal already has the old text", detail: "you typed in Terminal since"))
        expectFits(try await render(model, named: "moved-on"))
    }

    @Test func aLongTakeStaysWithinTheCard() async throws {
        let model = makeModel()
        let sentence = "Then put the fix in CLAUDE.md and run the tests again before the PR. "
        model.open(
            take(String(repeating: sentence, count: 12)), mode: .afterPaste, vocabulary: [])
        model.typed = "claude"
        expectFits(try await render(model, named: "long"))
    }

    // MARK: - The Lens as the dictation overlay

    @Test func listeningWithAPreview() async throws {
        let model = makeModel()
        model.listen(app: Self.terminal)
        model.holdHint = "⇧ to check before pasting"
        model.show(LivePreview(text: "Ask Claude why the server", catches: [], confirmedTokens: 3))
        expectFits(try await render(model, named: "listening"))
    }

    @Test func listeningBeforeTheFirstWords() async throws {
        let model = makeModel()
        model.listen(app: Self.terminal)
        expectFits(try await render(model, named: "listening-empty"))
    }

    @Test func landedWithASettledWord() async throws {
        let model = makeModel()
        model.listen(app: Self.terminal)
        model.show(LivePreview(text: "Ask Claude why this", catches: [], confirmedTokens: 2))
        model.finishing()
        model.land(
            DictatedTake(
                pairID: nil, text: "Ask Claude why the", catches: [], app: Self.terminal,
                pasted: true, pastedInto: Self.terminal))
        expectFits(try await render(model, named: "landed"))
    }

    @Test func aHeldTakeWaiting() async throws {
        let model = makeModel()
        model.open(
            DictatedTake(
                pairID: nil, text: "Ask cloud why.", catches: [], app: Self.terminal,
                pasted: false, held: true),
            mode: .held, vocabulary: [])
        expectFits(try await render(model, named: "held"))
    }
}
