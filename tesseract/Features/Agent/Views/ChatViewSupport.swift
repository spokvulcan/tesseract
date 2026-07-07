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

/// Line spacing of transcript prose. Textual's GitHub paragraph style renders
/// markdown at `.fontScaled(0.25)` — 0.25 × the font size, i.e. SwiftUI
/// `.lineSpacing(4)` at the transcript's 16pt. Every plain-`Text` render of
/// multi-line prose (the markdown-off mode, expanded thinking) must apply the
/// same number, or the two render modes read at visibly different line
/// heights.
let chatLineSpacing: CGFloat = chatBodyFontSize * 0.25

/// Metrics of the flat document transcript.
enum ChatLayout {
    /// Readable column width — content never stretches past this.
    static let columnMaxWidth: CGFloat = 720
    /// Width of the +/− collapse-marker slot in collapsible rows.
    static let markerWidth: CGFloat = 14
    /// The Row Rhythm: the one vertical spacing between transcript rows —
    /// between-item and within-message alike, no clustering exceptions. Every
    /// row stack must use this constant so the rhythm cannot drift.
    static let rowSpacing: CGFloat = 16
    /// Durations below this render no badge — "0.0s" reads as broken, and a
    /// near-instant operation has no cost worth flagging.
    static let minBadgeDuration: Duration = .milliseconds(100)
}

// MARK: - Blank parts

extension String {
    /// Edge-trim for multi-line transcript text rendered through a plain
    /// `Text`. The model brackets parts with cosmetic newlines ("\n\n" around
    /// think and tool blocks, a trailing "\n" before a block close); the
    /// markdown renderer collapses them, but `Text` draws each as a blank
    /// line — phantom vertical space that breaks the Row Rhythm. Interior
    /// formatting is untouched.
    var chatDisplayTrimmed: String {
        trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

extension ContentPart {
    /// Whether this part would render as an empty row: a whitespace-only text
    /// part — the model's cosmetic "\n\n" between tool calls. The stored
    /// message keeps them (state mirrors the event stream verbatim); the
    /// document views skip them so blank rows don't double the Row Rhythm.
    var isBlankRow: Bool {
        if case .text(let part) = self {
            return part.text.allSatisfy(\.isWhitespace)
        }
        return false
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
    /// Whole-second readout for the ticking Tool Clock: "4s", "1m 05s".
    /// Tenths would flicker at tick rate and read as anxiety, not
    /// information; the frozen badge (`chatBadge`) keeps them.
    var liveChatBadge: String {
        let totalSeconds = Int(components.seconds)
        if totalSeconds < 60 { return "\(totalSeconds)s" }
        return String(format: "%dm %02ds", totalSeconds / 60, totalSeconds % 60)
    }

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
