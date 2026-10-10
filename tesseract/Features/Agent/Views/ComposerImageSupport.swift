//
//  ComposerImageSupport.swift
//  tesseract
//
//  The image pieces both composers share — the agent chat's and Today's: a
//  pending image's thumbnail, the file picker, the drop target that takes an
//  image anywhere on the page, and the refresh that keeps Vision
//  Availability current. Each composer owns its own Composer Draft; these
//  read and write whichever one it hands them.
//

import SwiftUI
import UniformTypeIdentifiers

// MARK: - Picker

enum ImagePicker {
    /// Pick images from disk into `draft`, through the same funnel as paste
    /// and drop, so cap trims and unreadable files get the same notice
    /// (issue #167).
    static func pick(into draft: ComposerDraftController, message: String) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = ImageIngest.supportedUTTypes
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.message = message
        panel.begin { [weak draft] response in
            guard response == .OK else { return }
            let payload = PasteboardImageReader.ingest(fileURLs: panel.urls)
            DispatchQueue.main.async {
                draft?.handleGesture(payload)
            }
        }
    }
}

// MARK: - Page drop

extension View {
    /// Dropping an image anywhere on the page lands it in `draft`'s pending
    /// strip (slice #117), under an overlay saying so. `isTargeted` only flips
    /// for drags whose items conform to `.image`, so other drags never dim
    /// the page.
    func imageDropTarget(_ draft: ComposerDraftController, title: String) -> some View {
        modifier(ImageDropTarget(draft: draft, title: title))
    }

    /// Keeps a composer's Vision Availability current: re-probed on appear
    /// and whenever the selected model, its download or the vision setting
    /// changes — never per keystroke.
    func refreshesVisionAvailability(_ vision: VisionAvailabilityController) -> some View {
        modifier(VisionAvailabilityRefresh(vision: vision))
    }
}

private struct ImageDropTarget: ViewModifier {
    @Bindable var draft: ComposerDraftController
    let title: String

    func body(content: Content) -> some View {
        content
            .onDrop(of: [.image], isTargeted: $draft.isDropTargeted) { providers in
                draft.handleWindowImageDrop(providers)
            }
            .overlay {
                if draft.isDropTargeted {
                    ZStack {
                        Color.black.opacity(0.4)
                        VStack(spacing: 12) {
                            Image(systemName: "photo.badge.plus")
                                .font(.system(size: 44, weight: .light))
                            Text(title)
                                .font(.title2.weight(.medium))
                        }
                        .foregroundStyle(.white)
                        .padding(32)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
                    }
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
                    .transition(.opacity)
                }
            }
            .animation(.easeInOut(duration: 0.15), value: draft.isDropTargeted)
    }
}

private struct VisionAvailabilityRefresh: ViewModifier {
    let vision: VisionAvailabilityController
    @Environment(SettingsManager.self) private var settings
    @EnvironmentObject private var downloads: ModelDownloadManager

    func body(content: Content) -> some View {
        content
            .onChange(of: settings.selectedAgentModelID) { _, _ in vision.refresh() }
            .onChange(of: downloads.status(for: settings.selectedAgentModelID)) { _, _ in
                vision.refresh()
            }
            .onChange(of: settings.useVisionWhenAvailable) { _, _ in vision.refresh() }
            .onAppear { vision.refresh() }
    }
}

// MARK: - Thumbnail

/// An image as a square thumbnail: a pending one in a composer's strip,
/// clicked (not the ✕) to open full size in Quick Look (#116), or one in the
/// Jarvis panel. The ✕ never takes focus: the panel's buttons can't (its
/// macOS 27.0 focus freeze, `GlassPanel`).
struct ImageThumbnailView: View {
    let attachment: ImageAttachment
    var side: CGFloat = 56
    var onRemove: (() -> Void)?
    var onTap: (() -> Void)?

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Group {
                if let nsImage = NSImage(data: attachment.data) {
                    Image(nsImage: nsImage)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: side, height: side)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(.quaternary)
                        .frame(width: side, height: side)
                        .overlay {
                            Image(systemName: "photo")
                                .foregroundStyle(.secondary)
                        }
                }
            }
            .contentShape(RoundedRectangle(cornerRadius: 8))
            .onTapGesture { onTap?() }
            .help(onTap == nil ? "" : "Click to view full size")

            if let onRemove {
                Button(action: onRemove) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 16))
                        .foregroundStyle(.white, .black.opacity(0.6))
                }
                .buttonStyle(.plain)
                .focusable(false)
                .offset(x: 6, y: -6)
            }
        }
    }
}
