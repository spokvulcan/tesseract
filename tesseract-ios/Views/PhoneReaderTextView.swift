//
//  PhoneReaderTextView.swift
//  tesseract-ios
//
//  The Reader's text on the phone: a UIKit text view on TextKit 2, which
//  lays out only what is on screen, so a whole book opens and scrolls like a
//  page. The text is read-only. The heard word and sentence light up through
//  rendering attributes (no relayout), the view follows the voice unless you
//  scrolled a moment ago, and a tap jumps the reading to that sentence, or
//  moves the bookmark there at rest.
//

import Observation
import SwiftUI
import UIKit

struct PhoneReaderTextView: UIViewRepresentable {
    let reader: SpeechReader
    let font: UIFont
    let highlightStyle: ReadAlongHighlight
    /// Room under the text for the transport bar.
    let bottomInset: CGFloat

    func makeCoordinator() -> Coordinator { Coordinator(reader: reader) }

    func makeUIView(context: Context) -> UITextView {
        let textView = UITextView(usingTextLayoutManager: true)
        textView.isEditable = false
        textView.isSelectable = false
        textView.backgroundColor = .clear
        textView.alwaysBounceVertical = true
        textView.textContainerInset = UIEdgeInsets(top: 20, left: 20, bottom: 24, right: 20)
        textView.adjustsFontForContentSizeCategory = false
        textView.accessibilityLabel = "Text to read aloud"
        textView.delegate = context.coordinator
        let tap = UITapGestureRecognizer(
            target: context.coordinator, action: #selector(Coordinator.tapped(_:)))
        textView.addGestureRecognizer(tap)
        context.coordinator.attach(textView, font: font)
        return textView
    }

    func updateUIView(_ textView: UITextView, context: Context) {
        let coordinator = context.coordinator
        coordinator.setFont(font)
        coordinator.highlightStyle = highlightStyle
        if textView.contentInset.bottom != bottomInset {
            textView.contentInset.bottom = bottomInset
            textView.verticalScrollIndicatorInsets.bottom = bottomInset
        }
    }

    static func dismantleUIView(_ textView: UITextView, coordinator: Coordinator) {
        coordinator.detach()
    }

    // MARK: - Coordinator

    @MainActor
    final class Coordinator: NSObject, UITextViewDelegate {
        private let reader: SpeechReader
        private weak var textView: UITextView?
        private var font: UIFont?
        private var applied: SpeechReader.Highlight?
        private var lastUserScroll = Date.distantPast
        private var isObserving = false
        private var hasScrolledToBookmark = false
        var highlightStyle: ReadAlongHighlight = .both {
            didSet { if oldValue != highlightStyle { applyHighlight(force: true) } }
        }

        init(reader: SpeechReader) {
            self.reader = reader
        }

        func attach(_ textView: UITextView, font: UIFont) {
            self.textView = textView
            textView.text = reader.text
            setFont(font)
            observeReader()
            // Opens where reading resumes, once the view has a size.
            DispatchQueue.main.async { [weak self] in self?.scrollToBookmark() }
        }

        func detach() {
            isObserving = false
        }

        /// Twice: the first scroll lands on estimated heights for the text
        /// above, the second on its real layout.
        private func scrollToBookmark() {
            guard !hasScrolledToBookmark, reader.length > 0 else { return }
            hasScrolledToBookmark = true
            let range = NSRange(location: min(reader.bookmark, reader.length - 1), length: 1)
            scroll(to: range, force: true)
            DispatchQueue.main.async { [weak self] in self?.scroll(to: range, force: true) }
        }

        // MARK: Appearance

        func setFont(_ font: UIFont) {
            guard font != self.font, let textView else { return }
            self.font = font
            let paragraph = NSMutableParagraphStyle()
            paragraph.lineHeightMultiple = 1.25
            paragraph.paragraphSpacing = font.pointSize * 0.6
            let attributes: [NSAttributedString.Key: Any] = [
                .font: font, .foregroundColor: UIColor.label, .paragraphStyle: paragraph,
            ]
            textView.typingAttributes = attributes
            let storage = textView.textStorage
            if storage.length > 0 {
                storage.beginEditing()
                storage.setAttributes(
                    attributes, range: NSRange(location: 0, length: storage.length))
                storage.endEditing()
            }
            applyHighlight(force: true)
        }

        // MARK: Taps and scrolling

        @objc func tapped(_ gesture: UITapGestureRecognizer) {
            guard let textView,
                let position = textView.closestPosition(to: gesture.location(in: textView))
            else { return }
            reader.jump(to: textView.offset(from: textView.beginningOfDocument, to: position))
        }

        func scrollViewWillBeginDragging(_ scrollView: UIScrollView) {
            lastUserScroll = .now
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
                    changed = changed.union(frame(of: textRange, in: layoutManager))
                }
            }
            applied = next
            if let next {
                let accent = UIColor.tintColor
                if highlightStyle.showsSentence,
                    let sentence = textRange(next.sentence, in: layoutManager)
                {
                    layoutManager.addRenderingAttribute(
                        .backgroundColor, value: accent.withAlphaComponent(0.13), for: sentence)
                    changed = changed.union(frame(of: sentence, in: layoutManager))
                }
                if highlightStyle.showsWord, let word = textRange(next.word, in: layoutManager) {
                    layoutManager.addRenderingAttribute(
                        .backgroundColor, value: accent.withAlphaComponent(0.28), for: word)
                    layoutManager.addRenderingAttribute(.foregroundColor, value: accent, for: word)
                    changed = changed.union(frame(of: word, in: layoutManager))
                }
            }
            redraw(changed)
            if let next { scroll(to: next.word, force: false) }
        }

        /// TextKit 2 draws rendering attributes but doesn't redraw when they
        /// change: marks the views that draw the lines under `rect` (text
        /// container coordinates), a line or two per word, not the page.
        private func redraw(_ rect: CGRect) {
            guard let textView, !rect.isNull else { return }
            let lines = CGRect(
                x: textView.bounds.minX,
                y: rect.minY + textView.textContainerInset.top - 4,
                width: textView.bounds.width, height: rect.height + 8)
            var views = textView.subviews
            while let view = views.popLast() {
                let local = view.convert(lines, from: textView).intersection(view.bounds)
                guard !local.isNull, !local.isEmpty else { continue }
                view.setNeedsDisplay(local)
                views.append(contentsOf: view.subviews)
            }
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

        /// Keeps `range` in the comfortable upper middle of the view: only
        /// when it drifts out, and not for a few seconds after you scroll.
        private func scroll(to range: NSRange, force: Bool) {
            guard let textView, let layoutManager = textView.textLayoutManager,
                let textRange = textRange(range, in: layoutManager)
            else { return }
            guard force || Date.now.timeIntervalSince(lastUserScroll) > 4 else { return }
            layoutManager.ensureLayout(for: textRange)
            var rect = frame(of: textRange, in: layoutManager)
            guard !rect.isNull else { return }
            rect = rect.offsetBy(
                dx: textView.textContainerInset.left, dy: textView.textContainerInset.top)
            let inset = textView.adjustedContentInset
            let visible = CGRect(
                x: 0, y: textView.contentOffset.y + inset.top,
                width: textView.bounds.width,
                height: textView.bounds.height - inset.top - inset.bottom)
            let usable = CGRect(
                x: visible.minX, y: visible.minY + visible.height * 0.1, width: visible.width,
                height: visible.height * 0.6)
            guard force || !usable.contains(rect) else { return }
            let maxOffset = max(
                textView.contentSize.height + inset.bottom - textView.bounds.height, -inset.top)
            let target = min(
                max(rect.midY - visible.height * 0.35 - inset.top, -inset.top), maxOffset)
            textView.setContentOffset(CGPoint(x: 0, y: target), animated: !force)
        }
    }
}
