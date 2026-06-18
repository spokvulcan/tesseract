import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct AgentScrollableTextField: NSViewRepresentable {
    @Binding var text: String
    @Binding var dynamicHeight: CGFloat
    var onCommit: () -> Void
    var onImagePaste: (([ImageAttachment]) -> Void)?
    var isEnabled: Bool = true
    /// Return true if the arrow key was consumed (e.g. by popup navigation).
    var onArrowUp: (() -> Bool)?
    var onArrowDown: (() -> Bool)?
    var onEscape: (() -> Bool)?

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSTextView.scrollableTextView()
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.verticalScrollElasticity = .allowed
        scrollView.drawsBackground = false
        // Disable focus ring on scroll view
        scrollView.focusRingType = .none

        // Replace with our custom NSTextView that intercepts image paste
        // AppKit guarantees documentView is the NSTextView we installed.
        // swiftlint:disable:next force_cast
        let originalTextView = scrollView.documentView as! NSTextView
        let textView = ImagePasteTextView(frame: originalTextView.frame)
        textView.coordinator = context.coordinator
        scrollView.documentView = textView
        textView.delegate = context.coordinator
        textView.isRichText = false
        textView.drawsBackground = false
        textView.font = .systemFont(ofSize: 15)
        textView.textColor = .labelColor
        textView.isSelectable = true
        textView.isEditable = isEnabled
        textView.allowsUndo = true

        // Disable focus ring on text view
        textView.focusRingType = .none

        // Wrapping configuration
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.containerSize = NSSize(
            width: scrollView.contentSize.width, height: CGFloat.greatestFiniteMagnitude)
        textView.textContainer?.widthTracksTextView = true

        // Initial height calculation
        DispatchQueue.main.async {
            context.coordinator.recalculateHeight(textView: textView)
        }

        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        // AppKit guarantees documentView is the NSTextView we installed.
        // swiftlint:disable:next force_cast
        let textView = nsView.documentView as! NSTextView

        if textView.string != text {
            textView.string = text
            textView.setSelectedRange(NSRange(location: text.utf16.count, length: 0))
            context.coordinator.recalculateHeight(textView: textView)
        }

        // Disable focus ring
        nsView.focusRingType = .none
        textView.focusRingType = .none

        if textView.isEditable != isEnabled {
            textView.isEditable = isEnabled
            textView.textColor = isEnabled ? .labelColor : .secondaryLabelColor
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, NSTextViewDelegate {
        var parent: AgentScrollableTextField

        init(_ parent: AgentScrollableTextField) {
            self.parent = parent
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            self.parent.text = textView.string
            recalculateHeight(textView: textView)
        }

        func recalculateHeight(textView: NSTextView) {
            guard let layoutManager = textView.layoutManager,
                let textContainer = textView.textContainer
            else { return }

            layoutManager.ensureLayout(for: textContainer)
            let usedRect = layoutManager.usedRect(for: textContainer)

            // Set the new height (add a bit of padding to avoid clipping)
            let newHeight = usedRect.height

            DispatchQueue.main.async {
                if abs(self.parent.dynamicHeight - newHeight) > 1 {
                    self.parent.dynamicHeight = newHeight
                }
            }
        }

        func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            if commandSelector == #selector(NSResponder.insertNewline(_:)) {
                if NSApp.currentEvent?.modifierFlags.contains(.shift) == true {
                    textView.insertText("\n", replacementRange: textView.selectedRange())
                    return true
                } else {
                    // Force the text binding to update before sending
                    self.parent.text = textView.string
                    parent.onCommit()
                    return true
                }
            }
            if commandSelector == #selector(NSResponder.moveUp(_:)) {
                if parent.onArrowUp?() == true { return true }
            }
            if commandSelector == #selector(NSResponder.moveDown(_:)) {
                if parent.onArrowDown?() == true { return true }
            }
            if commandSelector == #selector(NSResponder.cancelOperation(_:)) {
                if parent.onEscape?() == true { return true }
            }
            return false
        }

        /// Reads any images from the pasteboard (the full PNG/JPEG/TIFF/GIF/WebP/
        /// HEIC set, plus a decoded `NSImage` or copied image file-URL) via the
        /// shared `ImageIngest` core and forwards them. Returns whether images were
        /// found; the caller always falls through to `super.paste` so a mixed
        /// text+image clipboard pastes its text too (slice #115).
        @discardableResult
        func handleImagePaste() -> Bool {
            guard let onImagePaste = parent.onImagePaste else { return false }
            let attachments = PasteboardImageReader.read(NSPasteboard.general)
            guard !attachments.isEmpty else { return false }
            onImagePaste(attachments)
            return true
        }
    }
}

// MARK: - ImagePasteTextView

/// Custom NSTextView that intercepts paste to also attach image content.
final class ImagePasteTextView: NSTextView {
    weak var coordinator: AgentScrollableTextField.Coordinator?

    override func paste(_ sender: Any?) {
        // Attach any images first, then always paste text. Image pasteboard data
        // does not materialize as text, so a mixed clipboard yields both (#115).
        coordinator?.handleImagePaste()
        super.paste(sender)
    }
}

// MARK: - Pasteboard Image Reader

/// The impure pasteboard edge for ⌘V (slice #115): pulls image content off an
/// `NSPasteboard` and funnels each candidate through `ImageIngest`. Tries, in
/// order, the richest source first: copied image *file* URLs (⌘C on a file in
/// Finder), raw image data of a supported type, then a decoded `NSImage` (copied
/// in Preview or a browser). The first source that yields attachments wins, so a
/// single image never double-attaches.
enum PasteboardImageReader {
    static func read(_ pasteboard: NSPasteboard) -> [ImageAttachment] {
        // 1. Image file URLs.
        let urlOptions: [NSPasteboard.ReadingOptionKey: Any] = [
            .urlReadingContentsConformToTypes: [UTType.image.identifier]
        ]
        if let urls = pasteboard.readObjects(forClasses: [NSURL.self], options: urlOptions)
            as? [URL],
            !urls.isEmpty
        {
            let attachments = urls.compactMap { url -> ImageAttachment? in
                guard let data = try? Data(contentsOf: url) else { return nil }
                let uti =
                    (try? url.resourceValues(forKeys: [.contentTypeKey]).contentType?.identifier)
                    ?? UTType(filenameExtension: url.pathExtension)?.identifier
                    ?? url.pathExtension
                return try? ImageIngest.ingest(
                    data: data, typeIdentifier: uti, filename: url.lastPathComponent
                ).get()
            }
            if !attachments.isEmpty { return attachments }
        }

        // 2. Raw image data of a supported type.
        for utType in ImageIngest.supportedUTTypes {
            let type = NSPasteboard.PasteboardType(utType.identifier)
            if let data = pasteboard.data(forType: type),
                let attachment = try? ImageIngest.ingest(
                    data: data, typeIdentifier: utType.identifier, filename: "pasted-image"
                ).get()
            {
                return [attachment]
            }
        }

        // 3. Decoded image with no file backing.
        if let image = NSImage(pasteboard: pasteboard),
            let attachment = try? ImageIngest.ingest(image: image, filename: "pasted-image.png")
                .get()
        {
            return [attachment]
        }

        return []
    }
}
