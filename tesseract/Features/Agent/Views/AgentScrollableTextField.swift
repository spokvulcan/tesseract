import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct AgentScrollableTextField: NSViewRepresentable {
    @Binding var text: String
    @Binding var dynamicHeight: CGFloat
    var onCommit: () -> Void
    /// Delivers every Image Gesture payload (paste or composer drag) — the
    /// receiver decides availability, capping, and feedback (issue #167).
    var onImageGesture: ((ImageGesturePayload) -> Void)?
    /// Mirrors an image-bearing drag hovering the composer, so the same
    /// full-window drop overlay shows over the text view too.
    var onImageDragTargeted: ((Bool) -> Void)?
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

        // Pick up the image/promise drag types `acceptableDragTypes` adds.
        textView.updateDragTypeRegistration()

        // Initial height calculation
        DispatchQueue.main.async {
            context.coordinator.recalculateHeight(textView: textView)
        }

        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        context.coordinator.parent = self

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

        /// The Image Gesture entry point (issue #167): if the pasteboard
        /// carries image content, the gesture resolves as an image action —
        /// read it (async only for file promises) and forward the payload —
        /// and the caller must NOT fall through to text insertion, even when
        /// the pasteboard also carries a textual sidecar (a copied file's
        /// name, a browser image's source URL) and even when nothing attaches
        /// (availability and rejection feedback belong to the receiver).
        /// Returns whether the gesture was claimed.
        func handleImageGesture(from pasteboard: NSPasteboard) -> Bool {
            guard parent.onImageGesture != nil else { return false }
            guard PasteboardImageReader.containsImageContent(pasteboard) else { return false }
            Task { @MainActor in
                let payload = await PasteboardImageReader.read(pasteboard)
                guard !payload.isEmpty else { return }
                parent.onImageGesture?(payload)
            }
            return true
        }

        func setImageDragTargeted(_ targeted: Bool) {
            parent.onImageDragTargeted?(targeted)
        }
    }
}

// MARK: - ImagePasteTextView

/// Custom NSTextView owning the composer's Image Gesture edges (issue #167):
/// image-wins paste, ⌘V enablement for image-only pasteboards, and drag
/// handling that mirrors the paste rule instead of inserting file paths.
final class ImagePasteTextView: NSTextView {
    weak var coordinator: AgentScrollableTextField.Coordinator?

    // MARK: Paste — image wins

    override func paste(_ sender: Any?) {
        if coordinator?.handleImageGesture(from: .general) == true { return }
        super.paste(sender)
    }

    override func pasteAsPlainText(_ sender: Any?) {
        if coordinator?.handleImageGesture(from: .general) == true { return }
        super.pasteAsPlainText(sender)
    }

    /// A plain-text NSTextView disables ⌘V when the pasteboard has no text
    /// representation, so a clipboard-only screenshot (⇧⌃⌘4) could never
    /// paste — enable the paste actions whenever the pasteboard carries image
    /// content.
    override func validateUserInterfaceItem(_ item: any NSValidatedUserInterfaceItem) -> Bool {
        if item.action == #selector(paste(_:)) || item.action == #selector(pasteAsPlainText(_:)),
            PasteboardImageReader.containsImageContent(.general)
        {
            return true
        }
        return super.validateUserInterfaceItem(item)
    }

    // MARK: Drag — mirrors the paste rule

    /// The image and file-promise types the stock plain-text view wouldn't
    /// accept; text and non-image file drags keep the default behavior.
    private static let imageDragTypes: [NSPasteboard.PasteboardType] =
        ImageIngest.supportedUTTypes.map { NSPasteboard.PasteboardType($0.identifier) }
        + NSFilePromiseReceiver.readableDraggedTypes.map { NSPasteboard.PasteboardType($0) }

    override var acceptableDragTypes: [NSPasteboard.PasteboardType] {
        super.acceptableDragTypes + Self.imageDragTypes
    }

    private func isImageGestureDrag(_ sender: any NSDraggingInfo) -> Bool {
        PasteboardImageReader.containsImageContent(sender.draggingPasteboard)
    }

    override func draggingEntered(_ sender: any NSDraggingInfo) -> NSDragOperation {
        if isImageGestureDrag(sender) {
            coordinator?.setImageDragTargeted(true)
            return .copy
        }
        return super.draggingEntered(sender)
    }

    override func draggingUpdated(_ sender: any NSDraggingInfo) -> NSDragOperation {
        if isImageGestureDrag(sender) { return .copy }
        return super.draggingUpdated(sender)
    }

    override func draggingExited(_ sender: (any NSDraggingInfo)?) {
        coordinator?.setImageDragTargeted(false)
        super.draggingExited(sender)
    }

    override func draggingEnded(_ sender: any NSDraggingInfo) {
        coordinator?.setImageDragTargeted(false)
        super.draggingEnded(sender)
    }

    override func prepareForDragOperation(_ sender: any NSDraggingInfo) -> Bool {
        if isImageGestureDrag(sender) { return true }
        return super.prepareForDragOperation(sender)
    }

    override func performDragOperation(_ sender: any NSDraggingInfo) -> Bool {
        if isImageGestureDrag(sender) {
            coordinator?.setImageDragTargeted(false)
            return coordinator?.handleImageGesture(from: sender.draggingPasteboard) == true
        }
        return super.performDragOperation(sender)
    }
}
