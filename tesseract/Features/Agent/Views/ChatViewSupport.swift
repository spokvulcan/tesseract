//
//  ChatViewSupport.swift
//  tesseract
//
//  Shared pieces of the chat transcript views: type metrics, the readable
//  column, and the async image thumbnail (Quick Look).
//

import SwiftUI

/// The one transcript type size — body, thinking, tool rows, badges, user
/// text all read at this size; hierarchy comes from color and weight, never
/// from size. The markdown renderer (Textual) derives all its font-scaled
/// metrics from the environment font, so applying
/// `.font(.system(size: chatBodyFontSize))` keeps every render mode identical.
let chatBodyFontSize: CGFloat = 16

/// Metrics of the flat document transcript.
enum ChatLayout {
    /// Readable column width — content never stretches past this.
    static let columnMaxWidth: CGFloat = 720
    /// Width of the +/− collapse-marker slot in collapsible rows.
    static let markerWidth: CGFloat = 14
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
