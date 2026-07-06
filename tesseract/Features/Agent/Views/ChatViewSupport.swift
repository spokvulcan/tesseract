//
//  ChatViewSupport.swift
//  tesseract
//
//  Shared pieces of the chat transcript views: type metrics, the readable
//  column, the copy affordance, and the async image thumbnail (Quick Look).
//

import SwiftUI

/// Base size for chat message body text. The markdown renderer (Textual)
/// derives all its font-scaled metrics from the environment font, so applying
/// `.font(.system(size: chatBodyFontSize))` keeps every render mode identical.
let chatBodyFontSize: CGFloat = 15

/// Metrics of the flat document transcript.
enum ChatLayout {
    /// Readable column width — content never stretches past this.
    static let columnMaxWidth: CGFloat = 720
    /// Secondary rows (tool calls, thinking) type size.
    static let stepFontSize: CGFloat = 12.5
}

// MARK: - Copy Button

struct ChatCopyButton: View {
    let text: String
    @State private var copied = false

    var body: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            withAnimation(.easeInOut(duration: 0.15)) { copied = true }
            Task {
                try? await Task.sleep(for: .seconds(1.2))
                withAnimation(.easeInOut(duration: 0.2)) { copied = false }
            }
        } label: {
            Image(systemName: copied ? "checkmark" : "doc.on.doc")
                .font(.system(size: 12))
                .foregroundStyle(copied ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
                .contentTransition(.symbolEffect(.replace))
        }
        .buttonStyle(.plain)
        .help("Copy")
    }
}

// MARK: - Async Image Attachment

/// Decodes image data off the main thread to avoid blocking scroll. Clicking
/// opens the image full size in Quick Look, navigable across the whole
/// conversation; the temp file is pre-warmed on decode so opening is
/// near-instant.
struct AsyncImageAttachmentView: View {
    let attachment: ImageAttachment
    @Environment(ComposerDraftController.self) private var composerDraft
    @State private var nsImage: NSImage?

    var body: some View {
        Group {
            if let nsImage {
                Image(nsImage: nsImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(.quaternary)
                    .overlay(ProgressView().controlSize(.small))
            }
        }
        .frame(maxWidth: 200, maxHeight: 200)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .contentShape(RoundedRectangle(cornerRadius: 8))
        .onTapGesture { composerDraft.openQuickLook(clicked: attachment.id) }
        .help("Click to view full size")
        .task(id: attachment.id) {
            let data = attachment.data
            nsImage = await Task.detached {
                NSImage(data: data)
            }.value
            composerDraft.prewarmImagePreview(attachment)
        }
    }
}

// MARK: - Duration formatting

extension Duration {
    /// Compact human duration for tool rows: "0.4s", "12s", "2m 05s".
    var chatBadge: String {
        let totalSeconds =
            Double(components.seconds)
            + Double(components.attoseconds) / 1e18
        if totalSeconds < 10 {
            return String(format: "%.1fs", totalSeconds)
        }
        if totalSeconds < 60 {
            return String(format: "%.0fs", totalSeconds)
        }
        let minutes = Int(totalSeconds) / 60
        let seconds = Int(totalSeconds) % 60
        return String(format: "%dm %02ds", minutes, seconds)
    }
}
