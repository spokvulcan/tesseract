//
//  ReaderTextView.swift
//  tesseract
//
//  The Reader's text: an AppKit text view on TextKit 2, which lays out only
//  what is on screen, so a whole book opens, scrolls and edits like a page.
//  At rest it is an editor. While reading it is read-only: the heard word
//  and sentence light up through rendering attributes (no relayout), the
//  view follows the voice unless you scrolled away a moment ago, a click
//  jumps the reading there, and Space pauses. "Read from Here" is always in
//  the context menu.
//

import AppKit
import Observation
import SwiftUI

struct ReaderTextView: NSViewRepresentable {
    let reader: SpeechReader
    let font: NSFont
    let highlightStyle: ReadAlongHighlight
    let isReading: Bool
    /// Room under the text for the floating control bar.
    let bottomInset: CGFloat

    /// The widest a line may run, however wide the window.
    static let columnWidth: CGFloat = 700

    func makeCoordinator() -> Coordinator { Coordinator(reader: reader) }

    func makeNSView(context: Context) -> ReaderScrollView {
        let textView = ReaderNSTextView(usingTextLayoutManager: true)
        textView.isRichText = false
        textView.importsGraphics = false
        textView.allowsUndo = true
        textView.drawsBackground = false
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.isContinuousSpellCheckingEnabled = false
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.textContainerInset = NSSize(width: 32, height: 28)
        textView.setAccessibilityLabel("Text to read aloud")

        let scrollView = ReaderScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.drawsBackground = false
        scrollView.automaticallyAdjustsContentInsets = false
        scrollView.documentView = textView

        context.coordinator.attach(textView, scrollView: scrollView, font: font)
        return scrollView
    }

    func updateNSView(_ scrollView: ReaderScrollView, context: Context) {
        let coordinator = context.coordinator
        coordinator.setFont(font)
        coordinator.setReading(isReading)
        coordinator.highlightStyle = highlightStyle
        if scrollView.contentInsets.bottom != bottomInset {
            scrollView.contentInsets = NSEdgeInsets(top: 0, left: 0, bottom: bottomInset, right: 0)
        }
    }

    static func dismantleNSView(_ scrollView: ReaderScrollView, coordinator: Coordinator) {
        coordinator.detach()
    }

    // MARK: - Coordinator

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate, NSTextStorageDelegate {
        private let reader: SpeechReader
        private weak var textView: ReaderNSTextView?
        private weak var scrollView: ReaderScrollView?
        private var font: NSFont?
        private var applied: SpeechReader.Highlight?
        private var lastUserScroll = Date.distantPast
        private var scrollObserver: NSObjectProtocol?
        private var isObserving = false
        var highlightStyle: ReadAlongHighlight = .both {
            didSet { if oldValue != highlightStyle { applyHighlight(force: true) } }
        }

        init(reader: SpeechReader) {
            self.reader = reader
        }

        func attach(_ textView: ReaderNSTextView, scrollView: ReaderScrollView, font: NSFont) {
            self.textView = textView
            self.scrollView = scrollView
            textView.string = reader.text
            setFont(font)
            textView.delegate = self
            textView.textStorage?.delegate = self
            textView.onJump = { [weak self] offset in self?.reader.read(from: offset) }
            textView.onReadFromHere = { [weak self] offset in self?.reader.read(from: offset) }
            textView.onTogglePause = { [weak self] in self?.reader.togglePause() }
            reader.liveText = { [weak textView] in textView?.string ?? "" }
            // Following the voice pauses for a few seconds after you scroll.
            scrollObserver = NotificationCenter.default.addObserver(
                forName: NSScrollView.willStartLiveScrollNotification, object: scrollView,
                queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.lastUserScroll = .now }
            }
            observeReader()
            scrollView.onFirstLayout = { [weak self] in self?.scrollToBookmark() }
        }

        /// Opens where reading resumes, once the view has a size. Twice: the
        /// first scroll lands on estimated heights for the text above, the
        /// second on its real layout.
        private func scrollToBookmark() {
            let length = reader.length
            guard length > 0 else { return }
            let range = NSRange(location: min(reader.bookmark, length - 1), length: 1)
            scroll(to: range, force: true)
            DispatchQueue.main.async { [weak self] in self?.scroll(to: range, force: true) }
        }

        func detach() {
            if let scrollObserver { NotificationCenter.default.removeObserver(scrollObserver) }
            scrollObserver = nil
            reader.textViewWillClose()
            isObserving = false
        }

        // MARK: Appearance

        func setFont(_ font: NSFont) {
            guard font != self.font, let textView else { return }
            self.font = font
            let paragraph = NSMutableParagraphStyle()
            paragraph.lineHeightMultiple = 1.28
            paragraph.paragraphSpacing = font.pointSize * 0.7
            let attributes: [NSAttributedString.Key: Any] = [
                .font: font, .foregroundColor: NSColor.labelColor, .paragraphStyle: paragraph,
            ]
            textView.font = font
            textView.defaultParagraphStyle = paragraph
            textView.typingAttributes = attributes
            if let storage = textView.textStorage, storage.length > 0 {
                storage.beginEditing()
                storage.setAttributes(
                    attributes, range: NSRange(location: 0, length: storage.length))
                storage.endEditing()
            }
        }

        func setReading(_ isReading: Bool) {
            guard let textView, textView.isReading != isReading else { return }
            textView.isReading = isReading
            textView.isEditable = !isReading
            textView.isSelectable = !isReading
            if isReading { textView.window?.makeFirstResponder(textView) }
            textView.window?.invalidateCursorRects(for: textView)
            applyHighlight(force: true)
        }

        // MARK: Following the reading

        private func observeReader() {
            isObserving = true
            withObservationTracking {
                _ = reader.highlight
            } onChange: { [weak self] in
                Task { @MainActor [weak self] in
                    guard let self, self.isObserving else { return }
                    self.applyHighlight(force: false)
                    self.observeReader()
                }
            }
        }

        private func applyHighlight(force: Bool) {
            guard let textView, let layoutManager = textView.textLayoutManager else { return }
            let next = reader.highlight
            guard force || next != applied else { return }
            var changed = CGRect.null
            if let old = applied {
                for range in [old.sentence, old.word] {
                    guard let textRange = textRange(range, in: layoutManager) else { continue }
                    layoutManager.removeRenderingAttribute(.backgroundColor, for: textRange)
                    layoutManager.removeRenderingAttribute(.foregroundColor, for: textRange)
                    changed = changed.union(visibleFrame(of: textRange, in: layoutManager))
                }
            }
            applied = next
            if let next {
                let accent = NSColor.controlAccentColor
                if highlightStyle.showsSentence,
                    let sentence = textRange(next.sentence, in: layoutManager)
                {
                    layoutManager.addRenderingAttribute(
                        .backgroundColor, value: accent.withAlphaComponent(0.13), for: sentence)
                    changed = changed.union(visibleFrame(of: sentence, in: layoutManager))
                }
                if highlightStyle.showsWord, let word = textRange(next.word, in: layoutManager) {
                    layoutManager.addRenderingAttribute(
                        .backgroundColor, value: accent.withAlphaComponent(0.28), for: word)
                    layoutManager.addRenderingAttribute(.foregroundColor, value: accent, for: word)
                    changed = changed.union(visibleFrame(of: word, in: layoutManager))
                }
            }
            redraw(changed)
            if let next { scroll(to: next.word, force: false) }
        }

        /// TextKit 2 draws rendering attributes but doesn't redraw when they
        /// change. Marks the views that draw the lines under `rect` (text
        /// view coordinates): a line or two per word, not the page.
        private func redraw(_ rect: CGRect) {
            guard let textView, !rect.isNull else { return }
            let lines = CGRect(
                x: textView.bounds.minX, y: rect.minY - 4, width: textView.bounds.width,
                height: rect.height + 8)
            var views = textView.subviews
            while let view = views.popLast() {
                let local = view.convert(lines, from: textView).intersection(view.bounds)
                guard !local.isNull, !local.isEmpty else { continue }
                view.setNeedsDisplay(local)
                views.append(contentsOf: view.subviews)
            }
        }

        /// Where `range` is drawn, in text view coordinates; null off screen,
        /// where no view draws it yet.
        private func visibleFrame(of range: NSTextRange, in layoutManager: NSTextLayoutManager)
            -> CGRect
        {
            guard let textView,
                let viewport = layoutManager.textViewportLayoutController.viewportRange,
                let shown = viewport.intersection(range)
            else { return .null }
            return frame(of: shown, in: layoutManager)
                .offsetBy(dx: textView.textContainerOrigin.x, dy: textView.textContainerOrigin.y)
        }

        /// The union of `range`'s line segments, in text container coordinates.
        private func frame(of range: NSTextRange, in layoutManager: NSTextLayoutManager) -> CGRect {
            var rect = CGRect.null
            layoutManager.enumerateTextSegments(in: range, type: .standard, options: []) {
                _, frame, _, _ in
                rect = rect.union(frame)
                return true
            }
            return rect
        }

        private func textRange(_ range: NSRange, in layoutManager: NSTextLayoutManager)
            -> NSTextRange?
        {
            guard let content = layoutManager.textContentManager,
                let start = content.location(
                    content.documentRange.location, offsetBy: range.location),
                let end = content.location(start, offsetBy: range.length)
            else { return nil }
            return NSTextRange(location: start, end: end)
        }

        /// Keeps `range` in the comfortable middle of the view: only when it
        /// drifts out, and not for a few seconds after you scroll yourself.
        private func scroll(to range: NSRange, force: Bool) {
            guard let textView, let scrollView, let layoutManager = textView.textLayoutManager,
                let textRange = textRange(range, in: layoutManager)
            else { return }
            guard force || Date.now.timeIntervalSince(lastUserScroll) > 4 else { return }
            layoutManager.ensureLayout(for: textRange)
            var rect = frame(of: textRange, in: layoutManager)
            guard !rect.isNull else { return }
            rect = rect.offsetBy(
                dx: textView.textContainerOrigin.x, dy: textView.textContainerOrigin.y)
            let visible = scrollView.documentVisibleRect
            let usable = CGRect(
                x: visible.minX, y: visible.minY + visible.height * 0.12, width: visible.width,
                height: visible.height * 0.62 - scrollView.contentInsets.bottom * 0.5)
            guard force || !usable.contains(rect) else { return }
            let target = CGPoint(x: 0, y: max(0, rect.midY - visible.height * 0.36))
            let clip = scrollView.contentView
            NSAnimationContext.runAnimationGroup { context in
                context.duration = force ? 0 : 0.35
                context.allowsImplicitAnimation = true
                clip.animator().setBoundsOrigin(target)
            }
            scrollView.reflectScrolledClipView(clip)
        }

        // MARK: Edits

        func textViewDidChangeSelection(_ notification: Notification) {
            guard let textView else { return }
            reader.selection = textView.selectedRange()
        }

        nonisolated func textStorage(
            _ textStorage: NSTextStorage, didProcessEditing editedMask: NSTextStorageEditActions,
            range editedRange: NSRange, changeInLength delta: Int
        ) {
            guard editedMask.contains(.editedCharacters) else { return }
            let length = textStorage.length
            MainActor.assumeIsolated {
                reader.textDidChange(edited: editedRange, delta: delta, newLength: length)
            }
        }
    }
}

/// Keeps the text in a readable column however wide the window is.
final class ReaderScrollView: NSScrollView {
    /// Runs once, after the first layout that gives the view a size.
    var onFirstLayout: (() -> Void)?

    override func layout() {
        super.layout()
        guard let textView = documentView as? NSTextView else { return }
        let side = max(32, (contentSize.width - ReaderTextView.columnWidth) / 2)
        if abs(textView.textContainerInset.width - side) > 0.5 {
            textView.textContainerInset = NSSize(width: side, height: 28)
        }
        if let onFirstLayout, contentSize.height > 0 {
            self.onFirstLayout = nil
            DispatchQueue.main.async(execute: onFirstLayout)
        }
    }
}

/// The text view: jumps, "Read from Here" and Space-to-pause while reading.
final class ReaderNSTextView: NSTextView {
    var isReading = false
    var onJump: ((Int) -> Void)?
    var onReadFromHere: ((Int) -> Void)?
    var onTogglePause: (() -> Void)?

    override func mouseDown(with event: NSEvent) {
        guard isReading, let onJump else {
            super.mouseDown(with: event)
            return
        }
        onJump(characterIndexForInsertion(at: convert(event.locationInWindow, from: nil)))
    }

    override func keyDown(with event: NSEvent) {
        let modifiers = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        if isReading, event.charactersIgnoringModifiers == " ", modifiers.isEmpty {
            onTogglePause?()
            return
        }
        super.keyDown(with: event)
    }

    override var acceptsFirstResponder: Bool { true }

    override func menu(for event: NSEvent) -> NSMenu? {
        let menu = super.menu(for: event) ?? NSMenu()
        let offset = characterIndexForInsertion(at: convert(event.locationInWindow, from: nil))
        let item = NSMenuItem(
            title: "Read from Here", action: #selector(readFromHere(_:)), keyEquivalent: "")
        item.target = self
        item.representedObject = offset
        menu.insertItem(item, at: 0)
        menu.insertItem(.separator(), at: 1)
        return menu
    }

    @objc private func readFromHere(_ item: NSMenuItem) {
        guard let offset = item.representedObject as? Int else { return }
        onReadFromHere?(offset)
    }

    override func resetCursorRects() {
        if isReading {
            addCursorRect(visibleRect, cursor: .pointingHand)
        } else {
            super.resetCursorRects()
        }
    }
}
