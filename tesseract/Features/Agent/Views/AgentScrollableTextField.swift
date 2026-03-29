import SwiftUI
import AppKit

struct AgentScrollableTextField: NSViewRepresentable {
    @Binding var text: String
    @Binding var dynamicHeight: CGFloat
    var onCommit: () -> Void
    var onImagePaste: (([ImageAttachment]) -> Void)?
    var isEnabled: Bool = true
    
    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSTextView.scrollableTextView()
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.verticalScrollElasticity = .allowed
        scrollView.drawsBackground = false
        // Disable focus ring on scroll view
        scrollView.focusRingType = .none

        // Replace with our custom NSTextView that intercepts image paste
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
        textView.textContainer?.containerSize = NSSize(width: scrollView.contentSize.width, height: CGFloat.greatestFiniteMagnitude)
        textView.textContainer?.widthTracksTextView = true
        
        // Initial height calculation
        DispatchQueue.main.async {
            context.coordinator.recalculateHeight(textView: textView)
        }
        
        return scrollView
    }
    
    func updateNSView(_ nsView: NSScrollView, context: Context) {
        let textView = nsView.documentView as! NSTextView
        
        if textView.string != text {
            // Preserve selection range if possible
            let selectedRange = textView.selectedRange()
            textView.string = text
            if selectedRange.location <= text.utf16.count {
                textView.setSelectedRange(selectedRange)
            }
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
                  let textContainer = textView.textContainer else { return }

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
            return false
        }

        /// Handles image data from the pasteboard, returning attachments if images were found.
        func handleImagePaste() -> Bool {
            guard let onImagePaste = parent.onImagePaste else { return false }

            let pb = NSPasteboard.general
            let imageTypes: [NSPasteboard.PasteboardType] = [.png, .tiff]
            guard pb.availableType(from: imageTypes) != nil else { return false }

            var attachments: [ImageAttachment] = []
            for type in imageTypes {
                if let data = pb.data(forType: type) {
                    let mimeType = type == .png ? "image/png" : "image/tiff"
                    attachments.append(ImageAttachment(data: data, mimeType: mimeType, filename: "pasted-image"))
                    break
                }
            }

            guard !attachments.isEmpty else { return false }
            onImagePaste(attachments)
            return true
        }
    }
}

// MARK: - ImagePasteTextView

/// Custom NSTextView that intercepts paste to handle image content.
final class ImagePasteTextView: NSTextView {
    weak var coordinator: AgentScrollableTextField.Coordinator?

    override func paste(_ sender: Any?) {
        if coordinator?.handleImagePaste() == true { return }
        super.paste(sender)
    }
}
